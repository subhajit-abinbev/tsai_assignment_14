# tsai_assignment_14

## Training Logs
step=10 avg_loss=10.9641 lr=1.5e-05
step=20 avg_loss=10.4309 lr=3e-05
step=30 avg_loss=9.4101 lr=4.5e-05
step=40 avg_loss=8.9130 lr=6e-05
step=50 avg_loss=8.5317 lr=7.5e-05
step=60 avg_loss=7.8691 lr=9e-05
step=70 avg_loss=7.3372 lr=0.000105
step=80 avg_loss=6.7988 lr=0.00012
step=90 avg_loss=6.4527 lr=0.000135
step=100 avg_loss=6.2413 lr=0.00015
step=250 avg_loss=5.2262 lr=0.000375
.
.
.
step=1000 avg_loss=4.3043 lr=0.0015
step=1500 avg_loss=4.3524 lr=0.00141667
step=2000 avg_loss=4.4966 lr=0.00133333
step=2500 avg_loss=4.9577 lr=0.00125
step=3000 avg_loss=5.1195 lr=0.00116667
.
.
.
step=4000 avg_loss=5.3564 lr=0.001
step=5000 avg_loss=5.4241 lr=0.000833333
step=6000 avg_loss=5.2177 lr=0.000666667
step=7000 avg_loss=5.1138 lr=0.0005
step=8000 avg_loss=4.9051 lr=0.000333333
step=9000 avg_loss=4.8386 lr=0.000166667
step=10000 avg_loss=4.8410 lr=0




## Top 5 Generations
[sample 1] prompt: In a quiet library,
In a quiet library,
He to be his death to the.

PETONTESEN:
How, nobleio, thou well
And my, their two, and be and
Theursely, likeier he:
Ay tells of the rest, I, now,
From I will, that, norIG you and ouron
And in the state of with a gentleman up,
How, I, for shall him to your,

St you have woo:
O his
----------------------------------------
----------------------------------------
[sample 2] prompt: In a quiet library,
In a quiet library,
And you home a faults I am, and no our at.
Where to be the queen, and be upon,
And I d us and'st.


DUKE:
I not her in the Lady high of my:
And my lord- one your are myself them,
And the fair, Cam have that I,
And was it holding; the cunning of a father of Luc,
I have, and be her hath'll,
And you
----------------------------------------
----------------------------------------
[sample 3] prompt: In a quiet library,
In a quiet library,
And a, and the queen's statue of blood:,
And, poorest you: but and heaven, the,
That she that of the



PETONTES:
That more, thereall I inain with thousand,
And is tost in all, I would,
And shall an, we hath thou
That dear, I shall, as for it leave.
KING RICH:
No, my lord,
And, if my lord
----------------------------------------
----------------------------------------
[sample 4] prompt: In a quiet library,
In a quiet library,
But I, old hath would man
B queen:
IERC, my lord, you a piece.
PA you might thatTI well,
And your me:
We not what is is my father, and a
This more that to getness to in brother,
That thyurse.


ServULIET:
O,

O, from
Whbal'd beTR.



GLOUCESTER:

Mst
----------------------------------------
----------------------------------------
[sample 5] prompt: In a quiet library,
In a quiet library,
Had shall thy not and, if with I'll to- fly
To, which change; and in the point of.




ESCALiteen says:
But I my lord of as I to theats.


PROSPERO:
I have leave to be?
BARIIO:
I have, but to my lord,
That you to you,-- king death
To, then.'. he let the king;
With


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