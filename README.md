# tsai_assignment_14

## Training Logs
- Placeholder: attach metrics and console logs from the 10,000-step DeepSeek run here.

## Top 5 Generations
- Placeholder: record the five favorite completions sampled after training.

## Model Architecture
- Token embedding layer followed by positional handling via rotary embeddings (RoPE).
- 24 decoder blocks, each composed of:
	- Multi-Head Latent Attention with 12 heads, 4 key/value heads, and 8 learnable latent tokens.
	- Mixture-of-Experts feed-forward layer with 8 experts (top-2 gating, jitter, and loss-free balancing).
	- Dual RMSNorm residual pathways for attention and feed-forward modules.
- Final RMSNorm and tied linear head for vocabulary projection.

### Detailed Architecture
1. **Embedding Block**
	 - Word embeddings: 49,152 vocabulary × 768 hidden size.
	 - Positional encodings: rotary embeddings parameterized by θ=10,000 with optional scaling/interleaving disabled.
2. **Decoder Stack (repeated 24×)**
	 - **Pre-Attention RMSNorm** with ε=1e-5.
	 - **Multi-Head Latent Attention**
		 - Query projection: 768 → 12×64.
		 - Key/Value projections: 768 → 4×64, expanded to 12 heads via grouping.
		 - Latent tokens: 8 learnable vectors per batch, concatenated ahead of sequence tokens.
		 - Rotary embedding applied to token positions (latents remain unrotated).
		 - Attention masking combines causal mask with token padding mask; dropout disabled.
		 - Output projection: 768 → 768.
	 - **Residual Add** back to input.
	 - **Post-Attention RMSNorm** with ε=1e-5.
	 - **Mixture-of-Experts Feed-Forward**
		 - Router: linear 768 → 8 logits with stochastic jitter.
		 - Top-2 gating with loss-free load balancing using running importance statistics.
		 - Experts (×8): linear 768 → 2048, SiLU activation, linear 2048 → 768, dropout 0.05.
		 - Aggregated expert outputs combined per token and per-slot weights.
	 - **Residual Add** back to attention output.
3. **Final RMSNorm** with ε=1e-5.
4. **Language Modeling Head** tying embeddings: linear 768 → 49,152, bias-free.