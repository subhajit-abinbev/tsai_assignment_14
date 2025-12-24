## Training Logs

| Step | Avg Loss | Learning Rate |
|-----:|---------:|---------------:|
| 10   | 10.9641  | 1.50e-05 |
| 20   | 10.4309  | 3.00e-05 |
| 30   | 9.4101   | 4.50e-05 |
| 40   | 8.9130   | 6.00e-05 |
| 50   | 8.5317   | 7.50e-05 |
| 60   | 7.8691   | 9.00e-05 |
| 70   | 7.3372   | 1.05e-04 |
| 80   | 6.7988   | 1.20e-04 |
| 90   | 6.4527   | 1.35e-04 |
| 100  | 6.2413   | 1.50e-04 |
| 250  | 5.2262   | 3.75e-04 |
| ⋮    | ⋮        | ⋮ |
| 1000 | 4.3043   | 1.50e-03 |
| 1500 | 4.3524   | 1.42e-03 |
| 2000 | 4.4966   | 1.33e-03 |
| 2500 | 4.9577   | 1.25e-03 |
| 3000 | 5.1195   | 1.17e-03 |
| ⋮    | ⋮        | ⋮ |
| 4000 | 5.3564   | 1.00e-03 |
| 5000 | 5.4241   | 8.33e-04 |
| 6000 | 5.2177   | 6.67e-04 |
| 7000 | 5.1138   | 5.00e-04 |
| 8000 | 4.9051   | 3.33e-04 |
| 9000 | 4.8386   | 1.67e-04 |
| 10000| 4.8410   | 0.00e+00 |

## Top 5 Generations

> **Sample 1** — Prompt: *In a quiet library,*
>
> In a quiet library,
> He to be his death to the.
>
> PETONTESEN:
> How, nobleio, thou well
> And my, their two, and be and
> Theursely, likeier he:
> Ay tells of the rest, I, now,
> From I will, that, norIG you and ouron
> And in the state of with a gentleman up,
> How, I, for shall him to your,
>
> St you have woo:
> O his

---

> **Sample 2** — Prompt: *In a quiet library,*
>
> In a quiet library,
> And you home a faults I am, and no our at.
> Where to be the queen, and be upon,
> And I d us and'st.
>
> DUKE:
> I not her in the Lady high of my:
> And my lord- one your are myself them,
> And the fair, Cam have that I,
> And was it holding; the cunning of a father of Luc,
> I have, and be her hath'll,
> And you

---

> **Sample 3** — Prompt: *In a quiet library,*
>
> In a quiet library,
> And a, and the queen's statue of blood:,
> And, poorest you: but and heaven, the,
> That she that of the
>
> PETONTES:
> That more, thereall I inain with thousand,
> And is tost in all, I would,
> And shall an, we hath thou
> That dear, I shall, as for it leave.
> KING RICH:
> No, my lord,
> And, if my lord

---

> **Sample 4** — Prompt: *In a quiet library,*
>
> In a quiet library,
> But I, old hath would man
> B queen:
> IERC, my lord, you a piece.
> PA you might thatTI well,
> And your me:
> We not what is is my father, and a
> This more that to getness to in brother,
> That thyurse.
>
> ServULIET:
> O,
>
> O, from
> Whbal'd beTR.
>
> GLOUCESTER:
>
> Mst

---

> **Sample 5** — Prompt: *In a quiet library,*
>
> In a quiet library,
> Had shall thy not and, if with I'll to- fly
> To, which change; and in the point of.
>
> ESCALiteen says:
> But I my lord of as I to theats.
>
> PROSPERO:
> I have leave to be?
> BARIIO:
> I have, but to my lord,
> That you to you,-- king death
> To, then.'. he let the king;
> With

## Model Architecture
- Token embedding layer followed by positional handling via rotary embeddings (RoPE).
- 24 decoder blocks, each composed of:
	- Multi-Head Latent Attention with 12 heads, 4 key/value heads, and 8 learnable latent tokens.
	- Mixture-of-Experts feed-forward layer with 8 experts (top-2 gating, jitter, and loss-free balancing).
	- Dual RMSNorm residual pathways for attention and feed-forward modules.
- Final RMSNorm and tied linear head for vocabulary projection.
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