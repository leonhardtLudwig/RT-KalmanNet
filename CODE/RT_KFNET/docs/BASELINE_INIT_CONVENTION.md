# Baseline correctness: the initialization-convention artifact (Task 4)

*Recorded 2026-07-21. Evidence from read-only scratchpad experiments; interpretation
from the REKF/UAVF + KalmanNet theory. Companion to [ORIGINAL_VS_MODIFIED.md](ORIGINAL_VS_MODIFIED.md).*

## Why this document exists

While validating the Task-4 baseline we found that, on the **matched** Synthetic-NL
model, the standalone EKF (−22.2 dB) beat the whole robust family (REKF/existing/
proposed ≈ −18.6 dB) by **3.6 dB**. That should be impossible: theory says
**REKF with `c=0` (⇒ `θ=0`, `V=P`) is *exactly* the EKF** (UAVF paper: *"if there is
no a-priori information take it equal to zero, i.e. compute the state prediction
using EKF"*; professor: *"if this is zero then you obtain exactly the standard EKF"*).
So a 3.6 dB gap flags a set-up error, not a real performance difference. This doc
records what it was and how to think about it.

## What we measured (scratchpad, read-only)

| Check | Result |
|---|---|
| #1 isolate `c` and `V0` | Gap survives `c→0` (θ≈0.9, ~0.1% inflation) **and** matched `V0`: REKF still −18.65 vs EKF −22.27. So it's **not robustness, not `V0`**. |
| #2 pinpoint | Per-timestep gap = +3.4/+5.1/+7.0/+8.6 dB at t=0..3, then **0.00 dB for t≥5**. Giving REKF one initial predict → **−22.273 == EKF −22.273 exactly**. |
| #3 burn-in drop | Dropping the first 5 steps: EKF −29.68, REKF −29.67, existing −29.51 → all collapse within 0.17 dB. |
| model-independence | Aligned-init REKF ≡ EKF on the **full** trajectory for Q=1e-3 (gap −0.002) and Q=1e-1 (gap +0.010). The "burn-in cutoff" needed instead **moves 6→2** with the model. |

## The cause (mechanism)

Two implementations, two **initialization conventions**, fed the same number `x0`:

- Standalone `ExtendedKalmanFilter`: **predict-then-update** → first prior `f(x0)`, cov `Q`.
- `RobustKalman` (REKF): **update-then-predict** → first prior `x0`, cov `V0`.

Same symbol `x0`, two meanings (state at `t=-1` vs a-priori estimate *of* `x0`), so the
first priors differ by one application of `f`. With `Q=1e-3` the filter converges in
~5 steps, so that single-step offset dominates the T=100 average. After t≈5 the two
filters are byte-identical. (The EKF only *looked* better because `f(zeros)` happened
to land nearer the true initial state than `zeros` — an accident of a bad shared init,
i.e. the signature of an artifact.)

## The conceptual reconciliation (the important part)

Separate two things that were being conflated:

- **(A) the recursion form** — *update-then-predict from a prior `(x̂0,V0)`* — is
  **intrinsic** to the REKF as the papers define it.
- **(B) the initial condition and its consistency across the filters compared** — a
  property of the *experimental set-up*, not of any algorithm.

**The predictor form is intrinsic, but it produces NO legitimate transient difference
between REKF(c=0) and EKF.** The identity `REKF(c=0)=EKF` is exact **at every timestep**
*given a shared initial condition*. So a correctly set-up comparison shows zero gap at
all steps; the transient we saw came purely from (B) — a convention mismatch — not (A).

### Why burn-in is the wrong fix (and is *not* principled)
1. **It masks a bug instead of fixing it.** A gap in the first steps means the two
   filters weren't initialized consistently. Deleting those steps hides the
   inconsistency rather than removing it.
2. **It would also erase real robustness behavior.** When comparing REKF(`c>0`) vs
   REKF(`c=0`)/EKF, any genuine early-time effect of robustness lives in exactly those
   first steps. Burn-in would delete the window where robustness is most likely to act.
3. **It's arbitrary / model-dependent.** The needed cutoff moved 6→2 when `Q` changed;
   a result that depends on an arbitrary cutoff is not defensible.

### The correct fix: align the initial prior
Give **every** filter one common prior `(x̂0, V0)` in the predictor-form convention the
papers specify (matching how the data's `x0` is generated), and evaluate over the
**entire** trajectory. Aligning does **not** modify the REKF algorithm — it makes the
**EKF baseline use the same convention** REKF already uses ("predict-first EKF" and
"update-first EKF" are the same filter; they differ only in what index `x0` is pinned
to). Then `REKF(c=0) ≡ EKF` exactly, and any `REKF(c>0)` behavior — early transient
included — is its true, uncompensated effect.

## Answers to the three questions
- **Predictor-form init: convention or intrinsic?** The *recursion* is intrinsic; the
  *numerical prior and its cross-filter consistency* is a set-up choice. Aligning fixes
  the EKF baseline's convention, not REKF's algorithm.
- **Is aligned init the only theoretically correct comparison?** Yes — `REKF(c=0)=EKF`
  is an all-timestep identity that holds only under a shared init; isolating the effect
  of `c` requires holding the init convention fixed.
- **Is the unaligned comparison ever valid?** Not as a comparison of the *algorithms* —
  it confounds `c` with the init convention (which we showed contributes the whole
  3.6 dB, robustness ≈ 0). Only "valid" as a black-box software-defaults comparison,
  which is not the estimation-performance question Task 4 asks.

## Evaluation protocol to use going forward
1. One shared prior `(x̂0, V0)` for **all** filters (EKF, REKF, existing, proposed,
   KalmanNet), in the predictor-form convention, matched to the data's `x0`.
2. Report over the **full** trajectory (no burn-in).
3. Expectation on matched data: **EKF ≈ REKF ≈ existing ≈ proposed** (robustness idle),
   with KalmanNet re-checked under the same aligned init (its full-traj lead was partly
   init-transient handling).
