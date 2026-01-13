# position_encoding_exp

Experiments to test whether **positional information can be recovered without explicit positional embeddings** in:

- **Unidirectional (causal) self-attention** vs
- **Bidirectional self-attention**

Motivated by “Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings” (DroPE), `2512.12167v1.pdf`.

## What the experiment tests

We train a small “vanilla” Transformer on a task where the **label depends on a token position**.

### Task: “argmax-token position”

- **Input**: a length-128 sequence of random token IDs in \([1, 128]\).
- **Label**: the **1-indexed position** of the **largest token ID** in the sequence.
  - Ties are broken by the *first occurrence*.

This avoids the trivial failure mode where labels are always \(1..128\) regardless of input.

### Why this is a good probe

- With **no positional embeddings** and **bidirectional** attention, the model is (approximately) permutation-invariant, so it can detect *what* the max token is but not *where* it was → accuracy should stay near **chance** (\(\approx 1/128\)).
- With **no positional embeddings** and **causal** attention, the mask itself breaks permutation symmetry and can enable implicit position coding (e.g., “how much context a token can see”) → accuracy can rise above chance as capacity increases.

We also include a “DroPE-like” condition where we **train with positions enabled, then drop them mid-training** and continue training briefly (“recalibration”).

## Quickstart (Windows)

Create an environment and install deps:

```powershell
chmod +x setup_env.sh
./setup_env.sh
source ~/env/bin/activate
```

Run a small sweep and generate plots:

```powershell
python -m src.experiment.run_sweep --out_dir outputs --device auto
```

Results:

- `outputs/results.csv`
- `outputs/plots/*.png`

