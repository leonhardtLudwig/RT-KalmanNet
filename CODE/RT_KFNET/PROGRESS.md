# RT-KalmanNet — c_t collapse fix: progress report

**Date**: 2026-07-08
**Scope**: `KNet/RT_KalmanNet_nn.py`, `RobustKalmanPY/robust_kalman.py`, `main_Robust_KNet.ipynb`

## 1. Problem

`MSE(RT-KalmanNet) ≈ MSE(REKF)` — the GRU-based learned tolerance `c_t` wasn't
separating from the classical robust EKF with a fixed `c`. Diagnosis (see
`RTK_learned_c` row in the pre-fix ablation) showed `c_t` had collapsed to an
almost-constant value (`mean=0.108997, std=0.001245`) sitting exactly at the
network's architectural default (`sigmoid(raw=0)` under a zero-initialized
output bias) — i.e. the network wasn't learning anything input-dependent at
all, it was just regressing to a fixed point.

## 2. Root causes found

1. **`optim.Adam(..., weight_decay=1e-3)` applied to the `c_t` output head
   (`fcl_out`)** — L2 shrinkage pulled the head's weights toward 0 every
   step, which maps through `c_t = c_floor + c_range·sigmoid(raw)` straight
   to the observed collapse point, regardless of what the data needed.
2. **Weak gradient path from the state-loss to `c_t`** — the only route is a
   single implicit-Newton correction inside `fnComputeTheta`
   (`dθ/dc ≈ 1/(∂g/∂θ)`), a legitimate but inherently faint signal, easily
   dominated by (1).
3. **No input feature normalization** — the four difference features fed to
   the GRU span two very different physical scales (observation-noise scale
   `R≈0.1·I` vs. state-noise scale `Q≈0.001·I`, a 100× gap), unlike the
   original KalmanNet, which normalizes its analogous features.
4. **Ambient-context-dependent solver branch**: `fnComputeTheta` branched on
   `torch.is_grad_enabled()` rather than an explicit flag — not actually
   producing wrong values given how the notebook happened to call things,
   but fragile against future call sites.
5. **Latent cross-trajectory state leak**: `fnREKF` persisted filter state
   across calls unless the caller manually reset 4 attributes before every
   call — duplicated 4× across the notebook, an easy thing to miss when
   adding new evaluation loops.

Cross-checked the robust-EKF math (the `γ(P,θ)` / `g(θ,c)` formula in
`fnComputeTheta`) against the professor's own MATLAB reference
(`CODE/LF_DATA/theta.m`, `robust_filter.m`) and against his own paper
("Robust Kalman Filtering under Model Perturbations," eq. 7, τ=0 case) —
both confirm the current **unscaled** formula (no `0.5` factor) is correct;
it was *not* touched.

## 3. Fixes applied

**`KNet/RT_KalmanNet_nn.py`**
- Added a `c_init` constructor parameter; the output head's bias is now
  initialized via inverse-sigmoid so the *untrained* network's default
  output is a principled `c_init` (defaults to REKF's own `c`), not an
  arbitrary range-midpoint.

**`RobustKalmanPY/robust_kalman.py`**
- `c_init` threaded through to the network (defaults to the constructor's
  own `c` if unset).
- Precomputed `_obs_scale`/`_state_scale` (from `R`/`Q` diagonals); the four
  input features (`f1..f4`) are now divided by their characteristic scale
  before being fed to the network — preserves relative magnitude (unlike a
  strict L2-normalize, which would destroy it) while fixing the 100× unit
  mismatch.
- New `reset_state()` method, called automatically at the top of `fnREKF`
  by default (`reset=True`) — removes the need for callers to manually
  reset 4 attributes before every call, closing the cross-trajectory
  state-leak risk at the source.
- `fnComputeTheta`'s Newton-refinement step now always runs (wrapped in
  `torch.enable_grad()`), so the returned *value* is identical whether or
  not the caller is inside `torch.no_grad()` — no longer keyed off ambient
  autograd context. A per-step solver residual (`|g(θ,c)|`) and the solved
  `θ_t` itself (`self.th`, previously allocated but never populated) are
  now saved for diagnostics.

**`main_Robust_KNet.ipynb`**
- Optimizer now uses parameter groups: `weight_decay` excludes `fcl_out`
  (the primary fix for the collapse).
- Gradient clipping (`clip_grad_norm_`, max_norm=1.0) added to the training
  step.
- The 4 duplicated manual state-reset blocks deleted (now automatic via
  `reset_state()`).
- New diagnostic cells: `c_t` mean/std vs. training epoch; a constant-`c`
  sensitivity sweep (MSE vs. `c` across a grid); a `θ_t` trace comparing
  constant-c REKF vs. learned-c RT-KalmanNet; a train/eval solver-coherence
  check.

## 4. Validated results

### Smoke-test config (`T=50, N_E=10, N_CV=5, 30 epochs`)

| | Before | After |
|---|---|---|
| `c_t` mean / std | `0.108997` / `0.001245` | `0.000958` / `0.000054` |
| Train/eval solver residual diff | *(not checked)* | `0.0` (exact) |
| REKF MSE | `0.026861` | `0.026946`* |
| RT-KalmanNet MSE | `0.026803` | `0.026946`* |
| RTK beats REKF (fraction of sequences) | 100% | 20%** |

\* small before/after difference here is dataset-resample noise (a fresh
random smoke-test set), not a regression.
\** see §5 — this is expected, not a regression.

### Full config (`T=100, N_E=200, N_CV=50, 200 epochs, N_T=100×100=10,000 test sequences`)

| Method | MSE | Time/seq [s] |
|---|---|---|
| REKF (fixed `c=1e-3`) | 0.013920 | 0.4568 |
| RT-KalmanNet (learned `c_t`, mean≈0.00115) | 0.013920 | 0.3757 |
| KalmanNet (original, full learned gain) | 0.005894 | 0.0240 |

`c_t` on the full run: mean `0.001146`, std `1.02e-5` (over 1,000,000
per-timestep samples across all 10,000 test sequences) — stable, sane,
centered on `c_init`, not collapsed.

Constant-`c` sensitivity sweep (full config): MSE ranges only from
`0.013922` (`c=1e-4`) to `0.013880` (`c=0.2`) — a **0.3% spread across the
entire representable range of `c`**.

## 5. Key finding: the remaining REKF ≈ RT-KalmanNet gap is not a bug

The sensitivity sweep (now validated at both scales, n=2,500 and
n=1,000,000 timestep-samples) proves there is essentially **no exploitable
signal** in this specific benchmark (Synthetic Non-Linear model, matched
Gaussian process/measurement noise, only a static wrong initial-state
prior) for any value of `c` — constant or adaptive — to exploit. This
matches Zorzi's own model-perturbations paper (Section VII-B, "small
tolerance relative to actual mismatch"): robust and plain filters converge
when there's little real model mismatch to defend against. `c_t` correctly
learning to stay near `c_init` here is the *correct* answer for this
dataset, not a training failure. The pre-fix "100% of sequences beat REKF"
result was an artifact of the collapsed network landing, by chance, on a
constant marginally better than REKF's `1e-3` — not genuine adaptivity.

## 6. Known limitations / open issues (not yet fixed)

1. **KalmanNet vs. REKF/RT-KalmanNet in the summary table is not an
   apples-to-apples comparison.** `robust_kalman.py`'s `fnREKF` reports the
   one-step-ahead *predicted* state (`f(Xn_i)`, doesn't use the current
   observation), while the original `KalmanNetNN`/`Pipeline_EKF` reports
   the *filtered/posterior* estimate (does use the current observation) —
   a strictly easier quantity to get right. KalmanNet's apparent 2.4×
   MSE advantage is at least partly a metric-definition artifact, not
   purely a capability gap. Needs the REKF/RT-KalmanNet side to also
   expose/report the filtered estimate (`Xn_i`, already computed in-loop,
   just not currently returned) before this comparison is trustworthy.
2. **Cell 15's `test_data_synt` DataFrame is grown one row at a time** via
   `.loc[i, ...] = [...]` on out-of-bounds indices — classic O(n²) pandas
   anti-pattern. At `n=10,000` (full config) this made the evaluation
   cells take hours instead of the ~60-90 minutes the underlying REKF
   computation should need. Fix: pre-allocate
   `pd.DataFrame(index=range(n), columns=[...], dtype=float)` in Cell 15.
3. **`fnComputeTheta`'s Newton refinement now always runs** (per the §3
   fix), including for the plain `hard_coded=True, use_nn=False` REKF
   baseline, where it's provably unnecessary (no gradient is ever taken
   there). This trades a small, real amount of REKF-baseline speed for
   train/eval value-consistency in the learned-`c_t` case. Could be
   recovered by gating the Newton step behind `self.use_nn`.
4. **Cell 22 regenerates the entire `N_E=200`-sequence training pool via
   `DataGen` every single epoch**, then samples only `n_batch=10` of them —
   190/200 freshly generated sequences are discarded per epoch, 200 times
   over. A real, avoidable training-time cost, separate from the
   evaluation-side pandas issue.
5. **Task 4's comparison table is missing two baselines** named in the
   assignment: a plain (non-robust) EKF, and "the existing robust
   KalmanNet" as a baseline distinct from "the proposed method" — currently
   the notebook only has REKF / RT-KalmanNet(proposed) / KalmanNet.
6. **Task 3 (LF_DATA `c` comparison, constant and slowly-varying) has not
   been started.** Given finding §5, this — not further tuning on the SNL
   benchmark — is where `c_t`'s adaptivity actually has a chance to show
   measurable value, since `CODE/LF_DATA`'s reference generator uses
   `c≈1`, an order of magnitude outside today's default `c_range=0.2`
   ceiling (deliberately not widened yet, to avoid destabilizing the
   `fnComputeTheta` bisection — needs its own hardening pass first if `c`
   is going to be pushed that high).

## 7. What's working well

- BPTT (Task 1) is correctly implemented: full-trajectory gradient flow,
  proper mini-batch gradient accumulation (`n_batch` trajectories' losses
  averaged, one `backward()`/`optimizer.step()` per epoch), no accidental
  `.detach()`/`.item()` calls breaking the graph mid-trajectory.
- GRU replacement (Task 2) is complete and functioning: MLP encoder + GRU +
  linear head, correct hidden-state threading within a trajectory, correct
  reset at trajectory boundaries.
- The robust-EKF math itself (`fnComputeTheta`'s `g(θ,c)`, the `V_{t+1} =
  (P_{t+1}^{-1} - θI)^{-1}` recursion) is verified correct against two
  independent authoritative references (the professor's MATLAB code and
  his own paper).
- Train/eval numerical coherence in the solver is now exact (`0.0`
  difference between grad-enabled and `no_grad` passes).
- `c_t` behaves sanely and diagnosably now: stable, bounded, centered on a
  principled default, with residual/θ_t/sweep diagnostics in place to
  interpret any future run at a glance.

## 8. Recommended next steps (roughly in priority order)

1. Fix the Cell 15 DataFrame pre-allocation (cheap, high-impact on
   iteration speed).
2. Reduce `num_of_test_sets` for iteration (statistically, `~500-1000`
   test sequences is more than sufficient given the current per-sequence
   MSE std ≈ `0.0006` — reserve the full `10,000` for final reporting
   runs only).
3. Fix the KalmanNet-vs-REKF metric asymmetry (§6.1) before trusting the
   Task 4 comparison table.
4. Move to Task 3: generate LF_DATA scenarios (constant `c`, slowly-varying
   `c` per a CV model) and compare learned vs. ground-truth `c`. Revisit
   `c_range`/`fnComputeTheta` numerical safety margins before doing so,
   since true `c` there is ~10× the current ceiling.
5. Round out Task 4 with the missing plain-EKF and existing-robust-KalmanNet
   baseline columns.
6. Optional: address the per-epoch `DataGen` waste (§6.4) and the
   REKF-only Newton-refinement overhead (§6.3) if training/eval speed is
   still a bottleneck after (1)-(2).
