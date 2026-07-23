# Task 4 — Design & Execution Document

*Canonical reference for the remainder of Task 4. Supersedes scattered notes.
Last consolidated 2026-07-23. Confidence tags used throughout:*
**[FACT]** *proven by experiment or by direct code reading ·*
**[INFER]** *reasonable inference from evidence ·*
**[HYP]** *hypothesis, not yet tested.*

> **Task 4 goal.** Compare five estimators — **EKF, Robust EKF (REKF), KalmanNet,
> existing Robust KalmanNet, proposed Robust KalmanNet** — on the Synthetic
> Non-Linear model (KalmanNet paper, Section IV-C), under **full** and **partial**
> information, reporting state MSE (dB) and runtime. The **acceptance criteria are
> the professor's four qualitative orderings** (Section 8).

---

## 1. Current understanding

### 1.1 Implementation status **[FACT]**
Nothing fundamental is missing. The Synthetic-NL model, `f`, `h`, the Jacobians, and
all five estimators exist and run:
- Model/params: `Simulations/Synthetic_NL_model/parameters.py`
  (`f(x)=α·sin(βx+φ)+δ`, `h(x)=a(bx+c)²`, `m=n=2`).
- EKF: `Filters/EKF.py` (self-contained autograd `getJacobian` — the Lorenz import
  was removed; Lorenz is out of scope) + `Filters/EKF_test.py`.
- REKF & proposed: `RobustKalmanPY/robust_kalman.py` (the "v3" fork) +
  `KNet/RT_KalmanNet_nn.py` (GRU head, `sigmoid → c∈(0,1)`).
- Existing Robust KalmanNet: frozen `robust_kalman_original.py` + `RT_KalmanNet_nn_original.py`.
- KalmanNet: `KNet/KalmanNet_nn.py` + `Pipelines/Pipeline_EKF.py`.
- Orchestration: `task4_comparison.ipynb`.
- A second fork exists (`*_leo.py`, `c = c_floor + c_range·sigmoid`) kept in reserve
  for the `c*>1` contingency (Section 4/7).

### 1.2 Paper fidelity — Section IV-C **[FACT, verified against the paper]**
Verified against the KalmanNet paper (arXiv 2107.10043, Section IV-C, retrieved via
the ar5iv HTML rendering). Our implementation matches on every specified item:

| Item | Paper (Sec IV-C, Table II) | Our setup |
|---|---|---|
| `f` (full) | α=0.9, β=1.1, φ=0.1π, δ=0.01 | identical |
| `h` | a=1, b=1, c=0 → `x²` | identical |
| Partial mismatch | change **`f` only**: α=1, β=1, φ=0, δ=0 | identical (`f_partial=sin(x)`) |
| Mismatch mechanism | data from **true** model; filters use **partial** | identical |
| Noise ratio | ν = q²/r² = **−20 dB** | identical |
| Trajectory length | T = 100 | identical |
| Dimensions | 2 × 2 | identical |
| KalmanNet training | on **true-model** data, MSE loss | identical |

The paper does **not** specify Robust EKF, Robust KalmanNet, the tolerance `c`, the
initial state, or dataset sizes. Those come from Zorzi's framework and the assignment.

### 1.3 Robust-filter literature **[FACT, verified against sources]**
From Zorzi (arXiv 1508.01906) and the nonlinear robust-filter line (arXiv 2506.04815):
`c` is a **fixed, user-specified tolerance** solving `γ(P,θ)=c`; typical experimental
operating point is **~0.1**; `c→0 ⇒ θ→0 ⇒ corrected covariance → nominal ⇒ the robust
filter coincides with the standard filter` (exact result quoted in the nonlinear paper).
Too-large `c` breaks contractivity (divergence). Our filter implements this framework
faithfully (`fnComputeTheta` solves `γ(P,θ)=c` by bisection, `V=(P⁻¹−θI)⁻¹`).

### 1.4 Experimentally confirmed **[FACT]**
- The **zero-init trap** (Section 2.1) — frozen partial-info estimate at `x0=0`.
- Off-trap, **REKF beats EKF** under the partial mismatch: MSE 0.129 → 0.033 as `c`
  grows 1e-4 → 1.0 (init `ones`, small sample).
- H1 (P-too-small) and H2 (broken θ-solver) **rejected**: `trace(V)` inflates to ~1e33;
  `θ` tracks `c` monotonically (0.37→4.57) with solver residual ~1e-6; the v3 solver has
  no `c`-independent fallback.
- `c≈2` diverges — consistent with the contractivity limit.

### 1.5 Confirmed by code inspection **[FACT]**
- **Dataset generation is decoupled from the filter/network run-time init** (Section 2.5).
- KalmanNet's init is `SysModel.m1x_0` read at run time; `NNTest` reuses `best-model.pt`.

### 1.6 Remaining hypotheses **[HYP]**
- Aligning the init makes **all four** orderings appear (only the EKF-pair, partial, is confirmed — small scale).
- **`RobustKalmanNet > KalmanNet` under partial info** (never tested; existential).
- The proposed GRU learns a **useful `c` off-trap** (never tested off-trap; weak-gradient concern from Task 3).
- Orderings are robust to init value / `c` / seed / FULL scale, and full-info orderings survive alignment.

---

## 2. Root-cause analysis

### 2.1 The zero-initialization trap **[FACT]**
The partial-information filters were initialized at `x0 = 0`. For the **partial** model:
- `f_partial(0) = sin(0) = 0` → `x=0` is a **fixed point** of the dynamics;
- `h(x)=x² ⇒ h'(x)=2x ⇒ h'(0)=0` → the observation Jacobian is the **zero matrix** at `x=0`.

The EKF/REKF Kalman gain is `L = V·Cᵀ·(C V Cᵀ + R)⁻¹`. With `C=0`, `L=0` **for any `V`**.
So the filter ignores observations, the estimate stays frozen at `0`, and the prediction
never leaves `0`. Result: the posterior is identically `0`, and `MSE = mean(target²)`,
**independent of `c`** — covariance inflation multiplies a zero Jacobian.
*Confirmed:* `‖Xn_out‖ = 0` exactly for all `c`; `trace(V)` simultaneously explodes to ~1e33.

### 2.2 Why only partial information is affected **[FACT]**
The trap requires the init to sit on the model's joint singularity. Under **full**
information `f_true(0) = 0.9·sin(0.1π)+0.01 ≈ 0.288 ≠ 0`, so the true dynamics push the
state off `x=0` immediately, `C≠0`, the gain is nonzero, and the filter operates
normally. The fixed point at `0` exists **only** for `f_partial(0)=0`.

### 2.3 Why EKF/REKF freeze but KalmanNet does not **[FACT, code + experiment]**
EKF/REKF compute the gain as `V·Cᵀ(...)` — a function of the **observation Jacobian**,
which is zero at `x=0`. KalmanNet computes the gain as a **neural-network output**
(`KalmanNet_nn.py`, `KGain_step`); it never forms `V·Cᵀ` and never uses `h'`. Therefore
`h'(0)=0` cannot zero KalmanNet's gain. KalmanNet at `x0=0` is merely *possibly
suboptimal in the transient*, not frozen (it achieved a sensible ~−12.8 dB while the
model-based filters returned the identical frozen value). This is why aligning the init
is the **trap fix** for EKF/REKF but only a **consistency refinement** for KalmanNet.

### 2.4 The initialization-convention mismatch **[FACT, code + notebook]**
The generative initial mean is `m1x_0 = ones` (`parameters.py`), and the data is
generated at `ones`. The notebook then overrides the **filter** init to `zeros`
(`sys_model.m1x_0 = torch.zeros(m,1)`), which is exactly the trap point for the partial
model. Thus the reported partial-info numbers (all model-based filters identical at
−1.206 dB) are the frozen-estimate signature, not a real comparison. **[INFER]** the
`zeros` override also introduced an *unintended* init mismatch beyond the paper's f-only
protocol.

### 2.5 Decoupling of dataset generation and run-time init **[FACT, code]**
`m1x_0` plays two independent roles:
- **(R1) data-generation seed** — `Extended_sysmdl.GenerateBatch` (`:176` uses
  `self.m1x_0`; `:206` seeds the generation loop). The produced `Input/Target` are then
  serialized by `DataGen` (`utils.py`), freezing the dataset at the generation-time init.
- **(R2) run-time init** — `Pipeline_EKF.NNTrain` (`:123-124`) and `NNTest` (`:302`) call
  `KNet.InitSequence(SysModel.m1x_0…)` with the *current* attribute; `NNTest` loads the
  saved weights (`:277`) and does not train.

Because the dataset is frozen at R1, **changing the filter/network run-time init after
generation cannot alter the data.** Consequently, aligning KalmanNet's init to the
existing data requires **only re-testing, not retraining** (Section 3.4). Full retraining
is mandatory *only* if the dataset is regenerated at a different generative init — which
the aligned-init fix, by construction, does not do.

---

## 3. Design decisions

### 3.1 Aligned initialization policy — filter init = generative `x0 = ones`
All five estimators, both scenarios, start at the generative init (`ones`). Rationale:
(a) it removes the degenerate `x=0` singularity that froze the model-based filters
(Section 2); (b) it matches the existing data, so no dataset regeneration is needed;
(c) it is faithful to the paper, whose partial-information mismatch is `f`-only — the
init is *known*, so the filter legitimately uses it. **[INFER]** any generic nonzero
init removes the trap; `ones` is the principled, paper-faithful choice. Open point to
confirm with the professor: whether the init is "known" (→ `ones`) or is itself part of
"partial information" (→ a nonzero *unknown* init; still not `zeros`).

### 3.2 Common initial covariance policy
Give every filter the **same** small nonzero initial covariance. Rationale: fairness —
currently EKF uses `sys.m2x_0` while REKF hardcodes `V_prev = 1e-3·I`, an inconsistency
unrelated to the trap. **[FACT]** the trap is `m1x_0`-only, so there is **no evidence
`m2x_0` must change to fix it**; the change is purely for cross-filter comparability, and
is done at the **notebook level** (override `rk.V_prev`, as Task 3 did) — *not* by editing
`robust_kalman.py`.

### 3.3 Reuse the existing dataset
The dataset was generated at `x0 = ones` and is decoupled from the run-time init
(Section 2.5). The aligned-init fix does not change the data, so the existing
train/CV/test tensors remain valid. Regenerating would be wasted computation and would
needlessly invalidate the KalmanNet checkpoint.

### 3.4 Re-test KalmanNet before retraining
Code proves a re-test at the aligned init is valid without retraining (Section 2.5): the
data is unchanged and `NNTest` runs the saved weights from the new init. The init affects
only the transient (KalmanNet's features are L2-normalized differences), so the re-tested
number is representative. Re-testing is seconds; retraining is a full run. **Therefore we
re-test first and retrain only if the re-test degrades materially** (train/test-init
consistency), reusing the same data.

### 3.5 Postpone proposed-network training
Proposed-net training is the most expensive step and carries the highest failure
probability (it depends on *two* unverified things: robustness beating KalmanNet **and**
the GRU learning a useful `c`). It is therefore deliberately gated behind the free
existential decision (Gate 2b). We never train the proposed net until we have shown the
target ordering is even reachable.

---

## 4. Remaining risks (ordered by importance)

| # | Risk | Why it matters | How it is tested | What would invalidate the plan |
|---|---|---|---|---|
| 1 | **`RobustKalmanNet > KalmanNet` (partial) may be unreachable** **[HYP]** | It is one of the four acceptance orderings; KalmanNet legitimately trains around the f-mismatch, so a robust EKF-based method may be unable to beat it. | Gate 2b: REKF-`c*` vs trained KalmanNet on the same partial test set — *before* any proposed training. | KalmanNet keeps winning clearly → the paper-faithful setup may not permit this ordering → **escalate to the professor**. |
| 2 | **Proposed GRU may not learn a useful `c` off-trap** **[HYP]** | Even if robustness *can* beat KalmanNet, the proposed method only wins if its GRU learns `c` near `c*`; Task 3 showed weak `c`-gradient, and this was never tested off-trap. | E4: inspect learned `c_t` vs `c*` and proposed MSE vs KalmanNet. | `c_t` stays ~0 / proposed ≤ KalmanNet → proposed loses; report honestly (architectural-only win). |
| 3 | **`c*` may exceed the v3 sigmoid ceiling (1)** **[HYP]** | The v3 proposed head is `sigmoid∈(0,1)`; if the beneficial tolerance is `>1`, the proposed net cannot represent it. | E1 records `c*`. | `c* > 1` → conditional net swap to the `leo`/`c_range` head at E4. |
| 4 | **Initialization sensitivity** **[HYP]** | If the orderings hold only for one specific nonzero init, the result is a knife-edge artifact. | E1 tests 2–3 nonzero inits. | Orderings flip across reasonable inits → not defensible. |
| 5 | **`c`-range (band) sensitivity** **[HYP]** | If `RobustEKF>EKF` holds only for a thin `c` band, the claim is fragile. | E1 sweeps `c`; report the width of the beneficial range. | Benefit only at a single `c` → weak claim. |
| 6 | **Scale sensitivity** **[HYP]** | Small-scale orderings may not persist at FULL scale/seeds. | E5 confirms at reporting scale. | FULL contradicts small-scale → investigate the scale factor. |
| 7 | **Checkpoint transfer** **[INFER]** | If the existing KalmanNet checkpoint does not transfer to the aligned init, one retrain is added. | E2 (Gate 2a). | Materially degraded → schedule E3 (bounded cost, one run). |
| 8 | **Full-info orderings may shift after alignment** **[INFER]** | `EKF≥RobustEKF`, `KalmanNet≥RobustKalmanNet` must still hold (robustness is a penalty on matched data). | E1 full-info arm. | Robust beats standard on matched data → setup error. |

---

## 5. Execution roadmap (gated)

For each: **objective · files · expected · decision · cancellation-if-gate-fails.**

### Step 0 — Init policy (decision, no experiment)
- **Objective:** set all filters, both scenarios, to `m1x_0 = ones` + a common small `m2x_0`.
- **Files:** `task4_comparison.ipynb` (the `sys_model.m1x_0 = zeros` line; notebook-level `rk.V_prev` override).
- **Expected:** configuration ready; no numbers yet.
- **Decision:** proceed to E1.

### E1 — EKF + REKF `c`-sweep @ aligned init, full + partial **[NO TRAINING]**
- **Objective:** validate the EKF-pair orderings, pick `c*`, confirm no trap, test 2–3 inits.
- **Files:** `task4_comparison.ipynb`. Read-only: `robust_kalman.py`, `Filters/EKF*.py`.
- **Expected:** full `EKF ≤ RobustEKF` (small gap, `REKF(c→0)≡EKF`); partial `RobustEKF(c*) < EKF` clear margin; `‖Xn‖>0`; stable across inits. Record `c*`.

### Gate 1 — do the EKF-pair orderings appear off-trap?
- **Pass** → E2 with `c*`.
- **Fail** (partial ordering absent at any `c`, or still frozen) → **STOP, re-diagnose**.
  **Cancels E2–E5.**
- Fragile across init/`c` → widen sweep, flag.

### E2 — Re-test existing KalmanNet checkpoint(s) @ aligned init **[NO TRAINING]**
- **Objective:** obtain KalmanNet's aligned-init number without training; decide if a retrain is needed.
- **Files:** `task4_comparison.ipynb` (`NNTest` on existing `best-model.pt`). First verify which checkpoint is on disk (full vs partial).
- **Expected:** MSE ~stable vs its zeros-init value (init affects only the transient; KalmanNet is not trapped).

### Gate 2a — does the checkpoint transfer?
- **Stable** → reusable; **skip E3**.
- **Degraded / no usable checkpoint** → schedule one retrain (E3).

### Gate 2b — **existential decision** (REKF-`c*` vs KalmanNet, same partial set) **[NO TRAINING if E2 stable]**
- `RobustEKF(c*) ≤ KalmanNet` → robust can beat KalmanNet → ordering plausible → **E4**.
- `KalmanNet wins clearly` → **ESCALATE to the professor** before any proposed training.
  **Cancels E4–E5** until resolved.

### E3 — Retrain KalmanNet @ aligned init **[1 RUN, only if Gate 2a degraded]**
- **Objective:** valid aligned-init KalmanNet baseline when the checkpoint doesn't transfer.
- **Files:** `task4_comparison.ipynb` — set init **before** `NNTrain`; **reuse the existing data (no `DataGen` re-run).**
- **Expected:** ~paper-level KalmanNet performance. Feeds Gate 2b.

### E4 — Train proposed Robust KalmanNet @ aligned init **[1 RUN, only if Gate 2b passes]**
- **Objective:** does the proposed method realize `RobustKalmanNet > KalmanNet`, and does its GRU learn a useful `c`?
- **Files:** `task4_comparison.ipynb`. Read-only: `robust_kalman.py`, `RT_KalmanNet_nn.py`.
  **Conditional:** if `c*>1`, swap to the `leo`/`c_range` head at the notebook level.
- **Expected:** `c_t` engages toward `c*`; proposed MSE < KalmanNet. Also run existing (untrained).

### Gate 3 — does the proposed method realize its ordering?
- **Pass** (proposed < KalmanNet and `c` engages) → all four orderings assembled → E5.
- **Fail** (`c`~0 / proposed ≤ KalmanNet) → **cancel E5 as a "win" claim**; report honestly
  (the EKF-pair ordering is already secured by E1); revisit `c_range`/objective.

### E5 — FULL-config final run **[EXPENSIVE, ONCE]**
- **Objective:** confirm all four orderings + runtimes at reporting scale.
- **Files:** `task4_comparison.ipynb` (`CONFIG='FULL'`, aligned init, `c*`, final table + runtime + honest conclusion).
- **Expected:** small-scale orderings persist with tighter margins.
- **Decision:** persist → Task 4 complete; contradict → investigate the scale factor.

---

## 6. Training budget

- **Best case: 0 training runs before the existential decision**, and **1 total** (proposed
  only) if the checkpoint transfers (E2 stable) and Gate 2b passes.
- **Worst case: 2 runs** (KalmanNet retrain at E3 + proposed at E4), plus the single FULL
  run at E5.
- **Each run's trigger:** E3 fires only if the checkpoint does not transfer (Gate 2a); E4
  fires only if the existential ordering is reachable (Gate 2b); E5 fires only if all four
  orderings hold at small scale (Gate 3).
- **Why training is delayed:** the two orderings involving KalmanNet are both unverified
  and are the ones with a theoretical reason to fail. The plan front-loads the free
  experiments (E1, E2) so the make-or-break existential question is answered with 0–1
  training runs, and the most failure-prone step (proposed training) is never spent until
  its target is shown reachable.

---

## 7. Practical implementation notes

- **Expected to be modified (documentation task excluded):** `task4_comparison.ipynb`
  only — the init line, the REKF/EKF `c`-sweep cell, the KalmanNet re-test/retrain calls,
  the notebook-level `rk.V_prev` override, and `CONFIG`.
- **Read-only (do not edit):** `RobustKalmanPY/robust_kalman.py`, `KNet/RT_KalmanNet_nn.py`,
  `KNet/KalmanNet_nn.py`, `Pipelines/Pipeline_EKF.py`, `Filters/EKF*.py`,
  `Simulations/*` (including `parameters.py`, `Extended_sysmdl.py`, `utils.py`), all
  `*_original.py` and `*_leo.py`, datasets (`*.pt`), and checkpoints — except as read at run time.
- **Conditional modification:** a **notebook-level** net swap to the `leo`/`c_range` head at
  E4, **only if** E1 finds `c* > 1` (the v3 sigmoid cannot represent it). Coordinate with
  Track-1 before adopting for the final deliverable.
- **No dataset regeneration** and **no `DataGen` re-run** are required (Section 3.3).
- All exploratory runs write to a scratch location, never the project tree.

---

## 8. Acceptance criteria

Task 4 is complete when, at the reported (FULL) scale and aligned init, **all four
qualitative orderings hold**:

- **Full information:** `EKF ≥ RobustEKF` (small penalty gap; `REKF(c→0) ≡ EKF`) and
  `KalmanNet ≥ RobustKalmanNet`.
- **Partial information:** `RobustEKF > EKF` (clear margin) and
  `RobustKalmanNet > KalmanNet` (clear margin; judged on the **proposed** net).

Plus the robustness checks:
- **No frozen filters** — `‖Xn_out‖ > 0` and sensible estimates in every cell (anti-trap check).
- **Fairness** — `REKF(c→0)` matches EKF at all timesteps under aligned init.
- **Stability** — the partial-info orderings hold across ≥2 reasonable nonzero inits and a
  non-trivial `c` band (not a knife-edge).
- **Honest reporting** — results tied to the four orderings and to the Zorzi-`c`
  literature; the existing (untrained) Robust KalmanNet may legitimately *not* beat
  KalmanNet and this is stated explicitly.

If `RobustKalmanNet > KalmanNet` proves unreachable in the paper-faithful setup, Task 4 is
**not** silently declared complete — the discrepancy is escalated to the professor with the
evidence.

---

## 9. Where to resume after an interruption

**Start at Step 0, then run E1** — the EKF + REKF `c`-sweep at the aligned init
(`x0 = ones`), both scenarios. It requires **no training**, is the cheapest experiment
with the highest information, and gates everything: it confirms (or refutes) the
initialization root-cause fix, yields `c*`, and verifies the two EKF-pair orderings. Do
**not** train anything until Gate 1 passes. Then run E2 (re-test the existing KalmanNet
checkpoint — still no training) and evaluate **Gate 2b**, the existential decision, which
is the single most important checkpoint of the entire task and is reachable with **zero to
one** training runs. Only after Gate 2b passes should the proposed network be trained (E4),
and only after Gate 3 should the expensive FULL run (E5) be executed.
