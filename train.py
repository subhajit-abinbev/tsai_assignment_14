import argparse
import contextlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from model import ModelBuildParams, build_model
from utils import (
    build_dataset,
    checkpoint_dir,
    generate_text,
    get_device,
    get_torch_dtype,
    load_config,
    load_training_state,
    make_dataloader,
    prepare_tokenizer,
    save_checkpoint,
    set_seed,
)


def create_optimizer(model: torch.nn.Module, optimizer_cfg: Dict[str, Any]) -> AdamW:
    lr = optimizer_cfg["learning_rate_scheduler"]["learning_rate"]
    beta1 = optimizer_cfg["optimizer_factory"].get("adam_beta1", 0.9)
    beta2 = optimizer_cfg["optimizer_factory"].get("adam_beta2", 0.95)
    eps = optimizer_cfg["optimizer_factory"].get("adam_eps", 1e-8)
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)
    kwargs: Dict[str, Any] = {
        "lr": lr,
        "betas": (beta1, beta2),
        "eps": eps,
        "weight_decay": weight_decay,
    }
    if optimizer_cfg["optimizer_factory"].get("torch_adam_is_fused", False) and torch.cuda.is_available():
        kwargs["fused"] = True
    try:
        return AdamW(model.parameters(), **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return AdamW(model.parameters(), **kwargs)


def create_scheduler(optimizer: AdamW, scheduler_cfg: Dict[str, Any], total_steps: int) -> Any:
    warmup_steps = min(scheduler_cfg.get("lr_warmup_steps", 0), total_steps)
    target_steps = max(total_steps, scheduler_cfg.get("lr_warmup_steps", 0) + 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=target_steps,
    )
    return scheduler


def run_training(
    model: torch.nn.Module,
    tokenizer: Any,
    dataloader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    scheduler: Any,
    device: torch.device,
    dtype: torch.dtype,
    start_step: int,
    train_steps: int,
    log_interval: int,
    eval_interval: int,
    eval_prompt: str,
    grad_clip: float,
) -> Tuple[int, float]:
    model.train()
    autocast_ok = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    running_loss = 0.0
    total_logged = 0
    current_step = start_step
    data_iter = iter(dataloader)

    while current_step < start_step + train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        current_step += 1
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        autocast_cm = (
            torch.autocast(device_type=device.type, dtype=dtype)
            if autocast_ok
            else contextlib.nullcontext()
        )

        with autocast_cm:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        loss.backward()
        if grad_clip:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        loss_value = loss.detach().float().item()
        running_loss += loss_value
        total_logged += 1

        if current_step % log_interval == 0:
            avg_loss = running_loss / total_logged
            lr = optimizer.param_groups[0]["lr"]
            print(f"step={current_step} avg_loss={avg_loss:.4f} lr={lr:.6g}")
            running_loss = 0.0
            total_logged = 0

        if eval_interval and current_step % eval_interval == 0:
            sample = generate_text(
                model,
                tokenizer,
                eval_prompt,
                device,
            )
            print("-" * 40)
            print(f"[step {current_step}] prompt: {eval_prompt}")
            print(sample)
            print("-" * 40)

    return current_step, running_loss / max(total_logged, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DeepSeek architecture on a local corpus")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data", type=Path, default=Path("input.txt"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--prompt", type=str, default="In a quiet library,")
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--resume-from", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["general"].get("seed", 42))
    device = get_device()
    dtype = get_torch_dtype(config["model"].get("dtype", "float32"), device)

    print(f"Using device={device} dtype={dtype}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"CUDA device: {props.name} ({total_gb:.2f} GB) compute capability {props.major}.{props.minor}")
        print(f"Current allocated: {torch.cuda.memory_allocated(device) / (1024 ** 2):.1f} MB")

    if device.type == "cuda":
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    tokenizer = prepare_tokenizer(
        config["tokenizer"]["tokenizer_name_or_path"],
        config["tokenizer"].get("tokenizer_revision"),
    )
    dataset = build_dataset(
        tokenizer,
        args.data,
        config["tokens"]["sequence_length"],
    )
    dataloader = make_dataloader(dataset, config["tokens"]["micro_batch_size"])

    build_params = ModelBuildParams(device=device, torch_dtype=dtype)
    start_step = 0
    resume_state: Optional[Dict[str, Any]] = None
    model_weights_path = None
    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise FileNotFoundError(f"Resume checkpoint {args.resume_from} does not exist")
        model_weights_path = str(args.resume_from)
        resume_state = load_training_state(args.resume_from)
        start_step = int(resume_state.get("step", 0))

    model = build_model(config, params=build_params, weights_path=model_weights_path)

    optimizer_cfg = config["optimizer"]
    optimizer = create_optimizer(model, optimizer_cfg)
    scheduler_steps = max(start_step + args.train_steps, optimizer_cfg["learning_rate_scheduler"].get("lr_warmup_steps", 0) + 1)
    scheduler = create_scheduler(
        optimizer,
        optimizer_cfg["learning_rate_scheduler"],
        scheduler_steps,
    )

    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        if "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])

    checkpoints_root = args.checkpoint_root
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    log_interval = config["logging"].get("iteration_step_info_interval", 10)
    log_interval = max(log_interval, 1)
    eval_interval = max(250, log_interval)
    grad_clip = optimizer_cfg.get("clip_grad", 0.0)

    final_step, final_avg_loss = run_training(
        model,
        tokenizer,
        dataloader,
        optimizer,
        scheduler,
        device,
        dtype,
        start_step=start_step,
        train_steps=args.train_steps,
        log_interval=log_interval,
        eval_interval=eval_interval,
        eval_prompt=args.prompt,
        grad_clip=grad_clip,
    )

    checkpoint_path = checkpoint_dir(checkpoints_root, final_step)
    save_checkpoint(
        checkpoint_path,
        model,
        tokenizer,
        optimizer,
        scheduler,
        final_step,
        extra={"phase": "single", "average_loss": final_avg_loss},
    )
    print(f"Saved checkpoint at {checkpoint_path}")
    print(f"Completed training through step {final_step}; next step will start at {final_step + 1}")

    print("Generating sample completions:")
    for sample_idx in range(1, 6):
        text = generate_text(
            model,
            tokenizer,
            args.prompt,
            device,
        )
        print("-" * 40)
        print(f"[sample {sample_idx}] prompt: {args.prompt}")
        print(text)
        print("-" * 40)

    metadata = {
        "device": str(device),
        "dtype": str(dtype),
        "start_step": start_step,
        "train_steps": args.train_steps,
        "final_step": final_step,
        "evaluation_prompt": args.prompt,
    }
    checkpoints_root.joinpath("run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
