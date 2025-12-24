from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

LOGGER = logging.getLogger(__name__)


torch.set_float32_matmul_precision("high")


@dataclass
class ModelBuildParams:
    device: Optional[torch.device] = None
    torch_dtype: Optional[torch.dtype] = None
    cache_dir: Optional[Path] = None


class DeepSeekConfig(PretrainedConfig):
    model_type = "deepseek_custom"

    def __init__(
        self,
        vocab_size: int = 49152,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 12,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Mapping[str, Any]] = None,
        rope_interleaved: bool = False,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        pad_token_id: Optional[int] = None,
        attention_bias: bool = False,
        num_latent_tokens: int = 8,
        latent_init_std: float = 0.02,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_router_jitter: float = 0.0,
        moe_load_balancing_decay: float = 0.9,
        moe_load_balancing_eps: float = 1e-2,
        ffn_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_interleaved = rope_interleaved
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.num_latent_tokens = num_latent_tokens
        self.latent_init_std = latent_init_std
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = max(1, min(moe_top_k, moe_num_experts))
        self.moe_router_jitter = moe_router_jitter
        self.moe_load_balancing_decay = moe_load_balancing_decay
        self.moe_load_balancing_eps = moe_load_balancing_eps
        self.ffn_dropout = ffn_dropout

    @classmethod
    def from_external_config(cls, payload: Mapping[str, Any]) -> "DeepSeekConfig":
        data = dict(payload)
        data.setdefault("model_type", cls.model_type)
        return cls(**data)


class DeepSeekRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance + self.eps)
        return normed.to(hidden_states.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class DeepSeekRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float, interleaved: bool) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.interleaved = interleaved
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cache", torch.empty(0), persistent=False)
        self.register_buffer("sin_cache", torch.empty(0), persistent=False)
        self._seq_len_cached = 0

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if seq_len <= self._seq_len_cached and self.cos_cache.device == device and self.cos_cache.dtype == dtype:
            return
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        if self.interleaved:
            freqs = torch.repeat_interleave(freqs, 2, dim=-1)
            cos, sin = freqs.cos(), freqs.sin()
        else:
            cos = freqs.cos()
            sin = freqs.sin()
            cos = torch.repeat_interleave(cos, 2, dim=-1)
            sin = torch.repeat_interleave(sin, 2, dim=-1)
        self.cos_cache = cos.to(dtype)
        self.sin_cache = sin.to(dtype)
        self._seq_len_cached = seq_len

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._set_cos_sin_cache(seq_len, device, dtype)
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]


def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if position_ids is None:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    else:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.latent_tokens = config.num_latent_tokens
        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = DeepSeekRotaryEmbedding(self.head_dim, config.rope_theta, config.rope_interleaved)
        self.attention_dropout = config.attention_dropout
        if self.latent_tokens > 0:
            latent = torch.randn(self.latent_tokens, self.hidden_size) * config.latent_init_std
            self.latent_embeddings = nn.Parameter(latent)
        else:
            self.register_parameter("latent_embeddings", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        if self.latent_embeddings is not None:
            latent_states = self.latent_embeddings.unsqueeze(0).expand(bsz, -1, -1).to(dtype=dtype)
            combined = torch.cat([latent_states, hidden_states], dim=1)
        else:
            combined = hidden_states
        total_len = combined.size(1)
        latent_len = total_len - seq_len

        query = self.q_proj(combined)
        key = self.k_proj(combined)
        value = self.v_proj(combined)

        query = query.view(bsz, total_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, total_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, total_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_kv_heads != self.num_heads:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)

        if seq_len > 0:
            cos, sin = self.rotary_emb(seq_len, device, dtype)
            token_query = query[:, :, latent_len:, :]
            token_key = key[:, :, latent_len:, :]
            token_query, token_key = apply_rotary(token_query, token_key, cos, sin, position_ids)
            query[:, :, latent_len:, :] = token_query
            key[:, :, latent_len:, :] = token_key

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)

        causal_mask = torch.ones(total_len, total_len, device=device, dtype=torch.bool).tril(diagonal=0)
        if latent_len > 0:
            causal_mask[:latent_len, :] = True
        key_padding = torch.ones(bsz, total_len, device=device, dtype=torch.bool)
        if attention_mask is not None and seq_len > 0:
            key_padding[:, latent_len:] = attention_mask.to(torch.bool)
        combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) & key_padding.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~combined_mask, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, total_len, self.hidden_size)

        attn_output = attn_output[:, latent_len:, :]
        return self.o_proj(attn_output)


class DeepSeekMoEFeedForward(nn.Module):
    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.router_noise = config.moe_router_jitter
        self.load_balancing_decay = config.moe_load_balancing_decay
        self.load_balancing_eps = config.moe_load_balancing_eps
        self.dropout = config.ffn_dropout
        self.activation_name = config.hidden_act.lower()
        if self.activation_name not in {"silu", "gelu"}:
            raise ValueError(f"Unsupported activation {config.hidden_act}")
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([self._build_expert() for _ in range(self.num_experts)])
        self.register_buffer(
            "running_importance",
            torch.ones(self.num_experts),
            persistent=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = hidden_states.shape
        flat_states = hidden_states.reshape(bsz * seq_len, hidden)
        router_logits = self.router(flat_states)
        if self.training and self.router_noise > 0.0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_noise
        router_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        with torch.no_grad():
            importance = router_probs.sum(dim=0).to(self.running_importance.dtype)
            weight = torch.as_tensor(
                1.0 - self.load_balancing_decay,
                device=flat_states.device,
                dtype=self.running_importance.dtype,
            )
            blended = torch.lerp(self.running_importance, importance, weight)
            self.running_importance.copy_(blended + self.load_balancing_eps)
        balance = self.running_importance[topk_indices]
        # Reweight expert probabilities to achieve loss-free load balancing.
        balanced = topk_probs / balance
        balanced = balanced / balanced.sum(dim=-1, keepdim=True)

        expert_outputs = torch.zeros(flat_states.size(0), self.top_k, hidden, device=flat_states.device, dtype=flat_states.dtype)
        for expert_id, expert in enumerate(self.experts):
            mask = topk_indices == expert_id
            if not mask.any():
                continue
            positions = mask.nonzero(as_tuple=False)
            token_idx = positions[:, 0]
            slot_idx = positions[:, 1]
            expert_in = flat_states.index_select(0, token_idx)
            expert_out = expert(expert_in).to(expert_outputs.dtype)
            expert_outputs[token_idx, slot_idx] = expert_out

        combined = (expert_outputs * balanced.unsqueeze(-1)).sum(dim=1)
        combined = F.dropout(combined, self.dropout, training=self.training)
        combined = combined.view(bsz, seq_len, hidden)
        return combined

    def _build_expert(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
            self._make_activation(),
            nn.Linear(self.intermediate_size, self.hidden_size, bias=False),
        )

    def _make_activation(self) -> nn.Module:
        if self.activation_name == "silu":
            return nn.SiLU()
        return nn.GELU()


class DeepSeekDecoderLayer(nn.Module):
    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadLatentAttention(config)
        self.mlp = DeepSeekMoEFeedForward(config)
        self.input_layernorm = DeepSeekRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = DeepSeekRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DeepSeekModel(nn.Module):
    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepSeekDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = DeepSeekRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> BaseModelOutputWithPast:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class DeepSeekForCausalLM(PreTrainedModel):
    config_class = DeepSeekConfig
    _tied_weights_keys = ["model.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__(config)
        self.model = DeepSeekModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        autocast_dtype = next(self.parameters()).dtype
        device_type = input_ids.device.type
        with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=device_type == "cuda"):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None, hidden_states=None, attentions=None)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def _tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.lm_head, self.model.embed_tokens)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


def _config_from_yaml(configuration: Mapping[str, Any]) -> DeepSeekConfig:
    model_section = configuration.get("model", {})
    model_cfg = dict(model_section.get("model_config", {}))
    general_cfg = configuration.get("general", {})

    defaults: Dict[str, Any] = {
        "vocab_size": 49152,
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "initializer_range": 0.02,
        "use_cache": True,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": model_cfg.get("pad_token_id"),
        "rope_interleaved": model_cfg.get("rope_interleaved", False),
        "rope_scaling": model_cfg.get("rope_scaling"),
        "attention_dropout": model_cfg.get("attention_dropout", 0.0),
        "pretraining_tp": model_cfg.get("pretraining_tp", 1),
        "num_latent_tokens": model_cfg.get("num_latent_tokens", 8),
        "latent_init_std": model_cfg.get("latent_init_std", 0.02),
        "moe_num_experts": model_cfg.get("moe_num_experts", 8),
        "moe_top_k": model_cfg.get("moe_top_k", 2),
        "moe_router_jitter": model_cfg.get("moe_router_jitter", 0.0),
        "moe_load_balancing_decay": model_cfg.get("moe_load_balancing_decay", 0.9),
        "moe_load_balancing_eps": model_cfg.get("moe_load_balancing_eps", 1e-2),
        "ffn_dropout": model_cfg.get("ffn_dropout", 0.0),
    }
    defaults.update(model_cfg)
    defaults.setdefault("seed", general_cfg.get("seed", 42))
    return DeepSeekConfig(**defaults)


def build_model(
    configuration: Mapping[str, Any],
    params: Optional[ModelBuildParams] = None,
    weights_path: Optional[str] = None,
) -> DeepSeekForCausalLM:
    params = params or ModelBuildParams()
    config = _config_from_yaml(configuration)
    torch_dtype = params.torch_dtype or torch.float32

    if weights_path is not None:
        model = DeepSeekForCausalLM.from_pretrained(
            weights_path,
            config=config,
            dtype=torch_dtype,
            device_map=None,
            cache_dir=str(params.cache_dir) if params.cache_dir else None,
        )
    else:
        model = DeepSeekForCausalLM(config)

    to_kwargs: Dict[str, Any] = {}
    if params.device is not None:
        to_kwargs["device"] = params.device
    if torch_dtype is not None:
        to_kwargs["dtype"] = torch_dtype
    if to_kwargs:
        model.to(**to_kwargs)
    return model


def download_pretrained_weights(
    configuration: Mapping[str, Any],
    destination: Path,
    params: Optional[ModelBuildParams] = None,
) -> Path:
    raise NotImplementedError(
        "Pretrained DeepSeek weights are not available for automatic download."
    )
