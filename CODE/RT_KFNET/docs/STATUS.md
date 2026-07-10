# Project Status — RT-KalmanNet

*Snapshot, updated 2026-07-10. Kept short on purpose — for the
detailed history see [CHANGELOG.md](CHANGELOG.md), [FINDINGS.md](FINDINGS.md),
[ROADMAP.md](ROADMAP.md).*

---

## What the project is (30-second version)

- The **EKF** is model-based: it needs an accurate model and known noise
  statistics. In the real world those are wrong or unknown → it degrades.
- The professor's **Robust EKF (REKF)** adds **one scalar knob `c`** (a
  "tolerance") that inflates the predicted covariance, making the filter
  distrust a wrong model. Problem: `c` must be tuned, and in reality it
  changes over time (e.g. GPS signal degrading).
- **KalmanNet** replaces the Kalman-gain computation with a neural network
  (no covariance, no known `Q`/`R`) — powerful but heavy.
- **Our job (RT-KalmanNet):** combine them — use a **GRU to learn `c_t`
  at every time step** instead of tuning it by hand. Target: **most
  accurate**, **2nd fastest** of all methods.

---

## Status against the 4 tasks

| # | Task (teacher's wording) | Status | Notes |
|---|---|---|---|
| **1** | Apply **BPTT** to train RT-KalmanNet (Sec. 3.D) | ✅ **Done** | Full closed-loop backprop through the whole T=100 trajectory + proper **mini-batch** training (replaced the single-trajectory training the professor's code had). Verified. |
| **2** | Replace the core net with a **GRU** | ✅ **Done** | Old dense DNN → encoder + **GRU** + output head producing `c_t`. Verified. |
| **3** | Compare **learned `c` vs. true `c`** on **LF_DATA** (c constant / c slowly-varying, CV model) | 🟡 **Milestone 1 done** | **Constant `c=1` recovered**: RT-KalmanNet learns `c_t → ~1` on the LFM data (MATLAB→Python bridge built; filter validated **bit-for-bit** vs the professor's MATLAB). Remaining: tighten the fit; multiple-`c` / slowly-varying-`c` cases (need regenerated data). |
| **4** | Compare **EKF / REKF / KalmanNet / existing-robust-KNet / proposed** on the nonlinear case (Sec. 4.C) | 🟡 **Partial** | REKF, KalmanNet, and our proposed method are wired up and compared **fairly** (both predicted & filtered MSE + run-times) on the synthetic nonlinear model. **Missing:** plain **EKF** (its file is currently broken) and the **existing robust KalmanNet** as a separate baseline. |

---

## What's working well 👍

- **Tasks 1 & 2 are solid and verified** — the two "engineering" deliverables are done.
- The whole implementation was **audited against both source papers** (KalmanNet + Zorzi's REKF) and is **faithful**: the REKF recursion, the `θ`-from-`c` solver (matches the professor's own MATLAB `theta.m` exactly), the input features, and the nonlinear model parameters are all correct.
- Fixed a real bug where the learned `c_t` had **collapsed to a constant** (it was being pulled to a default by weight-decay on the output head).
- **Big speed win**: one evaluation went from **7+ hours → a few minutes** (it was an accidental O(n²) data structure, not the math).
- Comparison is **fair**: we now report both the *predicted* and *filtered* error so REKF-family and KalmanNet are measured on the same footing.

## Where it stands now 👍/👎

**Update (2026-07-10): the `c`-learning now demonstrably works.** On the
**LF_DATA** benchmark (Task 3), which has a *genuine* worst-case model
mismatch, RT-KalmanNet learns the tolerance `c_t` up from a 0.15 start to the
neighborhood of the true **`c=1`** — far above the old `c_range=0.2` ceiling,
which *was* the blocker (now widened to 2). The learned filter's MSE is
competitive with the best hand-tuned constant `c`.

Two honest caveats:
- On the **matched SNL benchmark**, `c_t` stays flat — and that is *correct*:
  no model mismatch means robustness has no job. `c` only comes alive on
  mismatched data, so "RT-KalmanNet ≈ REKF on SNL" is expected, not a failure.
- On LF_DATA, `c_t` recovers the *neighborhood* of 1 but still wanders, because
  the error-vs-`c` landscape is shallow near the optimum (little gradient
  pressure to pin `c` exactly). A tighter fit is a tuning matter, not a bug.

---

## Next steps (in priority order)

1. **Tighten Task 3** — cleaner `c_t → 1` convergence (lower LR / more steps)
   and a matched RT-KalmanNet vs REKF-best-`c` comparison.
2. **Fold Task 3 into the notebook** (thread the `c_range` param) and do the
   *multiple-`c`* and *slowly-varying-`c`* scenarios (need regenerated LFM data).
3. **Finish Task 4** — add the plain **EKF** and the **existing robust
   KalmanNet** so the comparison table is complete.

---

*Bottom line: Tasks 1 & 2 (BPTT + GRU) are done; **Task 3's core is now
demonstrated** — the learned `c` recovers the true `c` on LF_DATA. Remaining
work is tightening Task 3 and completing Task 4's baseline table.*
