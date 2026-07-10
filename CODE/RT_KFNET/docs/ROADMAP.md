# Roadmap

Current status against the assignment's 4 tasks, and prioritized next
steps. Updated as decisions are made — this is the living document; see
[FINDINGS.md](FINDINGS.md) for the evidence behind each status, and
[CHANGELOG.md](CHANGELOG.md) for exact code changes.

## Task status

| Task | Status | Notes |
|---|---|---|
| **1. BPTT training** | ✅ Done, verified | Closed-loop, full-trajectory gradient flow, proper mini-batch gradient accumulation (`n_batch` trajectories averaged, one `backward()`/`optimizer.step()` per epoch). No further work planned. |
| **2. GRU replacement** | ✅ Done, verified | MLP encoder + GRU + linear head; correct hidden-state threading within a trajectory, correct reset at boundaries. No further work planned. |
| **3. LF_DATA `c` comparison** (constant / slowly-varying) | 🟡 Milestone 1 done | **Constant `c=1` recovered**: RT-KalmanNet learns `c_t → ~1` on the existing `data.mat` (filter validated vs MATLAB `VR/PR`; `Q=B·Bᵀ` fix; `c_range` widened to 2). Standalone `task3_lf_data.py` + `task3_c_recovery.png`. **Remaining**: fold into the notebook (thread `c_range` param), tighten `c_t` convergence, and the *multiple-constant-`c`* + *slowly-varying-`c`* scenarios (need regenerated LFM data — MATLAB or a numpy port; deferred). |
| **4. EKF/REKF/KalmanNet/existing-robust-KNet/proposed comparison** | 🟡 Partially done | REKF, RT-KalmanNet (proposed), KalmanNet are wired up and compared fairly (predicted + filtered metrics) on the full-info SNL benchmark. **Missing**: plain EKF baseline (`Filters/EKF.py` is currently broken, needs a fix first) and "existing robust KalmanNet" as a baseline distinct from the proposed method. |

## Why `c_t` hasn't visibly separated RT-KalmanNet from REKF yet

Two experiments now agree this isn't a training bug:
- **Full-information SNL benchmark** (Task 4's original scenario): no
  exploitable mismatch exists at all (sweep spread ~0.3%) — `c_t`
  correctly learns to stay near `c_init`.
- **Option B, partial-information SNL benchmark** (in progress): a real,
  severe, persistent mismatch exists (55× MSE degradation), but `c`
  *still* shows zero measured sensitivity (0.0% sweep spread) — likely
  because `c_range=0.2` doesn't reach the regime where `θ` responds,
  possibly compounded by a `c`-independent fallback branch in
  `fnComputeTheta`'s bisection. Not yet disambiguated. See
  [FINDINGS.md](FINDINGS.md#option-b--partial-information-experiment-2026-07-08).

**This is the current critical path**: until `c` can actually reach a
range where it demonstrably changes REKF's behavior, neither Option B nor
Task 3 (LF_DATA, which needs `c≈1`) can produce a meaningful result.

## Next steps, in priority order

1. **Widen `c_range` and re-sweep** (cheap — REKF-only, no retraining,
   ~5 minutes). Directly tests whether the current `0.2` ceiling is the
   reason for zero sensitivity in Option B. If sensitivity appears at
   larger `c`, that's the fix; if it stays flat even at e.g. `c=2`, the
   bisection fallback branch needs a real fix (add a proper residual/value
   check instead of the crude bracket-validity fallback).
2. **Re-run Option B's KalmanNet cell** (fix already applied, not yet
   re-verified — see CHANGELOG 2026-07-08). Expected to show KalmanNet
   handling the bias-only mismatch much better than REKF/RT-KalmanNet,
   since it has no `θ`/`c` ceiling at all — a good sanity check on the
   hypotheses above.
3. **Fix `Filters/EKF.py`'s missing import** (`Simulations.Lorenz_Atractor.parameters.getJacobian`
   doesn't exist) — needed before Task 4's plain-EKF baseline can run at
   all.
4. **Complete Task 4's comparison table** — add plain EKF (after #3) and
   "existing robust KalmanNet" as baselines distinct from the proposed
   method.
5. **Move to Task 3 (LF_DATA)** once `c_range`/`fnComputeTheta` are
   confirmed to work at larger `c` (from #1) — build the MATLAB→Python
   data bridge, generate constant-`c` and slowly-varying-`c` scenarios,
   compare learned vs. ground-truth `c`.
6. Lower priority / opportunistic: fix Cell 22's per-epoch `DataGen` waste
   (regenerates and mostly discards a fresh `N_E=200`-sequence pool every
   epoch) if training speed becomes a bottleneck.

## Decisions log

- **2026-07-08**: chose to pursue "Option B" (partial-info SNL benchmark)
  before "Option C" (Task 3 / LF_DATA), since B reuses the entire
  already-working Task 4 pipeline and is far cheaper to test than building
  a new LF_DATA bridge. Confirmed via a read-only Explore pass that this
  requires zero changes to `Extended_sysmdl.py`, `utils.py`,
  `robust_kalman.py`, `KNet/KalmanNet_nn.py`, or `Pipelines/Pipeline_EKF.py`
  — new notebook cells only.