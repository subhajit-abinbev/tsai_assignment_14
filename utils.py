import math
import random
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


@dataclass
class TrainingBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class PackedDataset(Dataset):
    """Packs a token stream into fixed-length causal LM samples."""

    def __init__(self, token_ids: torch.Tensor, sequence_length: int) -> None:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a 1D tensor")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        usable_tokens = (token_ids.numel() // sequence_length) * sequence_length
        if usable_tokens == 0:
            raise ValueError("Not enough tokens to form a single sample")
        trimmed = token_ids[:usable_tokens]
        self._samples = trimmed.view(-1, sequence_length)

    def __len__(self) -> int:
        return self._samples.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._samples[idx]
        return {
            "input_ids": sample.clone(),
            "attention_mask": torch.ones_like(sample),
            "labels": sample.clone(),
        }


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(explicit: Optional[str] = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_torch_dtype(dtype_like: Any, device: torch.device) -> torch.dtype:
    if isinstance(dtype_like, torch.dtype):
        dtype = dtype_like
    elif isinstance(dtype_like, str):
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        dtype = mapping.get(dtype_like.lower(), torch.float32)
    else:
        dtype = torch.float32
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def prepare_tokenizer(tokenizer_name: str, revision: Optional[str] = None) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataset(
    tokenizer: Any,
    text_path: Path,
    sequence_length: int,
    add_bos: bool = True,
    add_eos: bool = True,
) -> PackedDataset:
    with text_path.open("r", encoding="utf-8") as handle:
        corpus = handle.read()
    pieces: List[int] = []
    if add_bos and tokenizer.bos_token_id is not None:
        pieces.append(tokenizer.bos_token_id)
    tokenized = tokenizer(corpus, add_special_tokens=False, return_attention_mask=False)
    pieces.extend(tokenized["input_ids"])
    if add_eos and tokenizer.eos_token_id is not None:
        pieces.append(tokenizer.eos_token_id)
    token_tensor = torch.tensor(pieces, dtype=torch.long)
    return PackedDataset(token_tensor, sequence_length)


def make_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def checkpoint_dir(base_dir: Path, step: int) -> Path:
    path = base_dir / f"step-{step:05d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    directory: Path,
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    step: int,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    if tokenizer is not None:
        tokenizer.save_pretrained(directory)
    payload = {
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if extra:
        payload.update(dict(extra))
    torch.save(payload, directory / "training_state.pt")


def load_training_state(directory: Path) -> Dict[str, Any]:
    state_path = directory / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing training state at {state_path}")
    try:
        return torch.load(state_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(state_path, map_location="cpu")


def prediction_cycle(iterable: Iterable[Any]) -> Iterator[Any]:
    return cycle(iterable)


def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
) -> str:
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    generated = input_ids
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=generated, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            if temperature and temperature > 1e-5:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)
        attention_mask = torch.ones_like(generated, device=device)
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    model.train()
    return text
