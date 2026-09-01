# Flow-Matching Decoder

`D(z) -> x_hat` — reconstruct a video frame at pixel level **from the latent
tokens of a frozen world model alone.**

This is a **qualitative microscope for comparing frozen world-model latent
spaces**, not a generative product model. Sample sharpness and controlled
comparability are what it optimizes for; FID leaderboards are explicitly not the
goal. Everything upstream of it — every world model, and the VAE — is loaded,
`eval()`-ed and `requires_grad_(False)`-ed, and never trained here.

## What the decoder is given

By default (`model.frame_conditioning: none`) the latent tokens are the decoder's
**only** input. It never sees the frame it is reconstructing, nor the frame
before it, in training or at sampling time — so anything visible in a
reconstruction was carried by those tokens, because there was no other source.

The tokens are produced by a fixed protocol, logged on every cache run:

1. The **teacher (target) encoder** does the encoding — the EMA branch that
   produced the targets the world model's own loss was computed against. Not the
   context encoder, not the predictor.
2. It sees a **full training-length clip window** ending at the target step:
   `max(data.dataset_fpcs)` frames at the model's own `fps`, `tubelet_size` and
   `crop_size`. Every number comes from that model's pretraining config, so the
   encoder is never shown a clip shape it was not trained on.
3. Only the **last temporal step's** `S` patch tokens are handed to the decoder.
   Patch tokens are laid out `t * S + s`, so that is exactly temporal index
   `window_tokens - 1`.

Steps 2 and 3 together are why this differs from "embed one frame": the kept
tokens attended over the whole window, so the latent is *the last frame as the
world model represents it in the context of its clip*. This is index-for-index
what [generate_world_model_report.py](../../evals/generate_world_model_report.py)'s
`encode_ground_truth` does, so a latent cached here is the same tensor that report
scores.

Expect **markedly blurrier samples** than a frame-conditioned decoder produces.
That is the measurement, not a defect: a JEPA representation is trained for
prediction and discards appearance detail it does not need.

### The frame-conditioned variant

`model.frame_conditioning: current_frame` also feeds the codec latent of the
preceding frame, making the task `D(x_t, z) -> x_hat`. Much easier, much weaker
as a measurement — a sample can look plausible while barely reading the latent.
Kept because the *difference* between the two modes is itself informative, but
`none` is the default and the setting the deliverable is about.

The two modes give the patchify conv different input widths, so a checkpoint from
one **cannot** load into the other; it raises rather than loading wrongly. The
mode also lives in `model_config`, so the panel harness's audit refuses to
compare a `none` run against a `current_frame` one.

## Layout

| file | what it is |
|---|---|
| `src/models/flow_decoder/vae.py` | frozen pixel↔latent codec: diffusers `AutoencoderKL`, plus a weight-free `patchify` codec for offline runs and exact-roundtrip controls |
| `src/models/flow_decoder/latent_adapter.py` | normalization buffers, `d_m`→`cond_dim` projection, and the continuous coordinate encoding that absorbs varying token counts |
| `src/models/flow_decoder/blocks.py` | masked attention, AdaLN-Zero DiT block, Perceiver resampler, timestep/σ embeddings |
| `src/models/flow_decoder/decoder.py` | `FlowMatchingDecoder` — Config A (`perceiver`) and Config B (`direct`) |
| `src/models/flow_decoder/flow.py` | rectified-flow objective, deterministic ODE sampler, classifier-free guidance |
| `app/flow_decoder/latent_cache.py` | shard writer: runs the frozen world model, caches latents + frame pairs, fits normalization statistics |
| `app/flow_decoder/shard_dataset.py` | `ShardStream` (training) and `ShardSet` (random access for panels) |
| `app/flow_decoder/train.py` | training loop; fixed step budget, no early stopping |
| `evals/generate_flow_decoder_panels.py` | the six diagnostic panels + controlled-comparison audit + 2AFC pair list |
| `app/flow_decoder/app_2afc.py` | blinded forced-choice app |
| `configs/train/flow_decoder/config{A-perceiver,B-direct}.yaml` | the two conditioning routes |
| `tests/models/test_flow_decoder.py` | the contract: varying `L`/`d_m`, mask isolation, determinism, latent dropout |

## How varying latent shapes are handled

A world model hands over `z` of shape `(B, L, d_m)`, and all three numbers move
between models. Exactly three things are allowed to differ per world model, and
all three live in `LatentAdapter`:

1. **`d_m`** — absorbed by `nn.Linear(d_m, cond_dim)`, the only shape-varying
   parameter tensor in the model (`test_varying_width_needs_only_the_adapter`
   asserts it is the only one).
2. **grid shape** — absorbed by the **continuous coordinate encoding**. Every
   token carries `(t, y, x)` cell centres mapped onto the same `[-1, 1]` cube plus
   its own `(dt, dy, dx)` extent, expanded by a **fixed** Fourier bank. A 16×16
   and a 32×32 latent therefore land in one coordinate frame differing only in
   density. There are **no learned per-index positional embeddings** for latent
   tokens — a table cannot tell token 37 of a 16×16 grid from token 37 of a 32×32
   grid, and cannot be evaluated at all on a larger grid.
3. **normalization buffers** — fitted once by `latent_cache.py` over the training
   shards, installed as frozen buffers, never touched again.

`L` may also be ragged within a batch: pass `z_mask`, and padded tokens are
provably invisible (`test_padding_does_not_change_the_output`).

## Pipeline

### 0. Environment

```bash
pip install diffusers          # frozen KL VAE
pip install gradio             # only for the 2AFC app
```

On this machine `diffusers` needs conda's newer `libstdc++` on the loader path
(`scipy` fails to import otherwise — a pre-existing env quirk unrelated to this
code):

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

To skip the VAE entirely (no download, no `diffusers`), set
`codec.kind: patchify` — a lossless space-to-depth codec. Good for plumbing and
roundtrip controls, not for the comparison panels.

### 1. Cache latents, once per world model

Enable the world model in `evals/generate_patch_embedding_report.py`'s
`MODELS_CONFIG`, then write **disjoint** train and eval shard sets:

```bash
python -m app.flow_decoder.latent_cache \
    --model-name "V-JEPA2 (baseline)" \
    --output-dir data/flow_decoder_shards/vjepa2-baseline/train \
    --dataset-csv data/ek55_4fps_train.csv \
    --num-clips 512 --targets-per-clip 6

python -m app.flow_decoder.latent_cache \
    --model-name "V-JEPA2 (baseline)" \
    --output-dir data/flow_decoder_shards/vjepa2-baseline/eval \
    --dataset-csv data/ek55_4fps_test.csv \
    --num-clips 64 --targets-per-clip 6 --store-predictor-latents --seed 777
```

A cached sample is `x_t` (last frame of temporal token *j−1*), `x_{t+1}` (last
frame of token *j*), `z_target` (the target encoder's tokens for *j*, embedded
inside a full-length window exactly as `encode_ground_truth` does it) and
optionally `z_pred` (the predictor's teacher-forced one-step prediction of *j*).
`x_t → x_{t+1}` is one autoregressive step of the world model.

Codec latents are deliberately **not** cached: the VAE is frozen and swappable
and costs a millisecond, whereas re-running a ViT-L world model costs tens of
minutes. Keeping pixels in the shard means the codec can change without touching
the expensive half.

Use the **same `--seed`, `--dataset-csv` and `--targets-per-clip`** for every
world model — the panels index eval samples positionally, so identical settings
are what makes "same clip, every row" true.

### 2. Train, once per world model

The **same config file** trains a decoder for any world model; only the shard set
changes.

```bash
python -m app.flow_decoder.train \
    --config configs/train/flow_decoder/configA-perceiver.yaml \
    --override data.train_shards=data/flow_decoder_shards/vjepa2-baseline/train \
    --override data.eval_shards=data/flow_decoder_shards/vjepa2-baseline/eval \
    --override folder=preliminary_experiments/flow-decoder/vjepa2-baseline-configA
```

- **Fixed step budget, no early stopping.** No code path can end a run early. The
  eval loss is logged for monitoring and nothing reads it. Stopping one run where
  *its* loss bottomed out would make training length a function of the latent
  space, confounding every panel.
- **No per-world-model tuning.** Any config difference between world models other
  than `d_m`, grid shape and normalization buffers is a bug by the spec's own
  definition, and step 3 enforces it.
- Objective: rectified flow, MSE on the velocity `x_1 - eps`, logit-normal `tau`.
  **No GAN loss, no perceptual loss, no discriminator** — adversarial sharpness
  is indistinguishable from detail the latent actually carried, which would
  destroy the one property the panels depend on.
- `flow.latent_dropout` (0.1) trains the null-memory branch that CFG needs;
  `flow.sigma_max` (0.5) adds noise to the *normalized* latent with σ fed in as a
  conditioning scalar, so the decoder learns the whole σ family at once and can
  be asked for σ=0 at sampling time. That buys robustness to off-distribution
  rollout latents without blurring the clean case.
- `x_t` is **never** dropped — only `z` is. That is what makes the guidance scale
  a measurement of the latent alone.

### 3. Generate the panels

```bash
python evals/generate_flow_decoder_panels.py \
    --decoder "V-JEPA2 (baseline)=preliminary_experiments/flow-decoder/vjepa2-baseline-configA/final.pt" \
    --decoder "High V-JEPA=preliminary_experiments/flow-decoder/high-vjepa-configA/final.pt" \
    --eval-shards "V-JEPA2 (baseline)=data/flow_decoder_shards/vjepa2-baseline/eval" \
    --eval-shards "High V-JEPA=data/flow_decoder_shards/high-vjepa/eval" \
    --output-dir preliminary_experiments/evals/vitl/flow_decoder_panels
```

Every checkpoint is audited against the first: `model_config`, `flow_config`, the
fixed step budget, the codec identity and the latent source must match exactly;
only `latent_dim` and `latent_grid` may differ. **A mismatch aborts the report**
rather than producing a picture that looks like a finding.

| panel | question |
|---|---|
| 1 `reconstruction` | What does the latent pin down? `x_t` (context) \| truth \| **codec ceiling** \| `D(z_target)` \| `D(z_pred)` |
| 2 `guidance` | What does the latent add over persistence? CFG sweep `w = 0 … 5` |
| 3 `seeds` | What does the latent leave free? Same latent, K seeds, + per-pixel std map |
| 4 `lead_time` | How does legibility decay with distance? Teacher-forced ladder, or caller-supplied rollout latents |
| 5 `token_ablation` | Which pixels does which token block govern? |
| 6 `crossmodel` | The comparison itself, + a scored **latent-swap control** (`follows`) |

Two failure modes look like findings and are not, and the report says so:

1. A blurry sample whose **codec ceiling** neighbour is also blurry — that is the
   frozen VAE, and it says nothing about the world model. Always read panel 1
   against the ceiling column, never against the truth.
2. A panel-6 **`follows`** figure at or below 50% — the fraction of latent-swapped
   samples that land closer to the donor clip's true frame than to their own
   column's. At chance, the decoder is not tracking its latent and every other
   panel for that run is void. Under `none` this failure looks like plausible but
   *generic* reconstructions (the same scene whatever the latent); under
   `current_frame` it looks like near-copies of `x_t`.
3. Blur that is **uniform across every model** compared — a property of this
   decoder configuration and step budget, not of any one latent space.

Rollout latents for panel 4 are **supplied by the caller**; this harness never
rolls a world model forward (that is `evals/generate_world_model_report.py`'s
job, and multi-frame rollout generation is out of scope). Expected file:

```python
torch.save({"latents": (K, S, d_m), "frame_prev": (3, H, W) uint8,
            "truth": (K, 3, H, W) uint8}, path)   # `truth` optional
```

### 4. Blinded forced choice

```bash
python -m app.flow_decoder.app_2afc \
    --pairs preliminary_experiments/evals/vitl/flow_decoder_panels/pairs_2afc.json
```

Model names never reach the browser; side assignment is seeded per trial (stable
across a reload, uncorrelated with identity) and trial order is shuffled per
session. Responses append to `responses.jsonl` with a running tally and an exact
two-sided sign test — a forced-choice preference is only readable with its count
attached.

## Out of scope

The ℓ2-regression Probe decoder (separate deliverable), training or modifying any
world model, training or fine-tuning the VAE, GAN/perceptual losses and
discriminators, multi-frame autoregressive rollout *generation*, and any web UI
beyond the 2AFC app above.
