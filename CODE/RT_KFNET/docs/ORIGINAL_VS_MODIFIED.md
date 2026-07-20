# Original vs Modified — what we changed, and whether it's correct

*Verified 2026-07-20 against the current repo working tree and the professor's
original files (`robust_kalman.py`, `RT_KalmanNet_nn.py`,
`main_Robust_KNet_original.ipynb`). Companion to [CHANGELOG.md](CHANGELOG.md) /
[FINDINGS.md](FINDINGS.md).*

## Purpose

The assignment is to **improve the existing robust KalmanNet** by (1) applying
BPTT, (2) replacing the core net with a GRU, (3) comparing learned vs. true `c`,
(4) comparing all filters. This document is the **line-by-line diff between the
professor's original code and our modified code**, with a correctness verdict on
each change, so the team can see what we did right and what still needs attention.

Legend: ✅ correct / justified · ⚠️ correct but with a caveat · ❌ regression or bug.

---

## TL;DR

- ✅ **Task 1 (BPTT)** and ✅ **Task 2 (GRU)** are implemented correctly, across
  all three files (network, filter, and training loop).
- The original network is **structurally untrainable**: we verified that a
  state-MSE `backward()` leaves **all 6 net parameters with `grad=None`**
  (total grad-norm `0`). Our changes are exactly what fix that.
- ⚠️ **Main thing to reconsider:** the modified `c`-head caps `c` at
  `≈0.20`, which is *lower* than the original's reach of `≈1.0`. That cap is the
  likely reason `c` shows no leverage under mismatch and can't reach Task-3's
  `c≈1`.
- ⚠️ Minor: a `c`-independent fallback branch in `fnComputeTheta`, and `fnREKF`
  no longer returning `V`.

---

## 1. `KNet/RT_KalmanNet_nn.py` — the network (Tasks 1 & 2)

### Side by side

| Aspect | Original (teacher) | Modified (ours) | Verdict |
|---|---|---|---|
| Core architecture | FCL + dense **DNN-with-feedback** (`nn.ModuleList` of `Linear`) | MLP encoder (`Linear→LayerNorm→GELU→Linear→GELU`) + **`nn.GRU`** + linear head | ✅ Task 2 |
| Recurrence | `self.previous_output` fed back, **detached** every step | GRU hidden state `h_t` returned and threaded (gradient flows) | ✅ Task 1 |
| `c` head | `sigmoid` → **`c ∈ (0, 1)`** | `c_floor + c_range·sigmoid` → **`c ∈ [1e-4, 0.2004]`** | ❌ regression (see below) |
| Init | default | Xavier + inverse-sigmoid bias so untrained `c ≈ c_init` | ✅ anti-collapse |
| `forward` signature | `forward(x)` → `c` | `forward(x, h_t)` → `(c, h_new)` | ✅ needed for GRU |

### The one line that mattered most (Task 1)

Original, at the end of `forward`:

```python
self.previous_output = final_output.clone().detach().requires_grad_(False)
```

This **severs the temporal gradient**: the feedback the net sees at step *t* has
no gradient path back to step *t−1*, so BPTT through time is impossible. The
modified net removes this by carrying recurrence in the **GRU hidden state**,
whose gradient is preserved across steps. This single change is the heart of
Task 1 on the network side.

### The regression to reconsider (`c` range)

- Original: `sigmoid(...)` → `c` can be anywhere in **(0, 1)**.
- Modified: `c_floor + c_range·sigmoid(...)` with `c_floor=1e-4, c_range=0.2`
  → `c` is capped at **≈0.2004**.

The paper's grid search and our own docs put the useful tolerance up to `c≈1`
(and Task 3's least-favorable data needs `c≈1`). So the modified head, while more
stable, **can't reach the regime the problem actually needs** — the original
head could. **Fix:** widen `c_range` (≈1.0–2.0). ⚠️/❌

---

## 2. `RobustKalmanPY/robust_kalman.py` — the filter

| Aspect | Original | Modified | Verdict |
|---|---|---|---|
| `fnREKF` forward | in-place writes to persistent buffers (`self.Xrekf[:,i+1]=…`, `self.C[:,:,i]=…`) | **local state variables + `torch.stack`** at the end | ✅ required for BPTT |
| `c_array` | `self.c.item()` — a **detached float** | keeps the **tensor** (gradient-carrying) | ✅ required to learn `c` |
| `fnComputeTheta` | plain bisection (`while`), **non-differentiable in `c`** | bisection (no_grad) + **implicit-Newton refinement** giving `dθ/dc` | ✅ essential |
| Input features F1–F4 | raw | scaled by `√diag(R)` / `√diag(Q)` | ✅ fixes ~100× scale gap |
| Posterior estimate | not exposed | `Xn_out` (filtered `x_{i|i}`) | ✅ fair MSE vs KalmanNet |
| State reset | inline in `__init__` only | `reset_state()` + detached truncated-BPTT boundary | ✅ correct truncated BPTT |
| Numerics | `torch.linalg.eig`, `log(det)` | `eigvalsh`, `slogdet`, symmetrization | ✅ more robust |
| θ refinement gating | always | only when `use_nn=True` (plain REKF skips it) | ✅ perf |

### Why the original couldn't learn `c` — two independent reasons

1. **Detached feedback** (network, §1) breaks BPTT across time.
2. **Non-differentiable θ solve**: the original `fnComputeTheta` selects `θ` by
   control-flow comparisons (`if value > 0: t2 = theta …`). The returned `θ` is a
   bracket midpoint, not a differentiable function of `c`, so **`dθ/dc ≈ 0`**.
   Even within a single step, `c → θ → loss` carries no gradient.

Together these guarantee zero gradient to the net — which we confirmed empirically
(`grad=None` on all params). The modified filter fixes reason 2 with the
**implicit function theorem**: around the bisection root it computes
`dθ/dc = 1 / (∂g/∂θ)` via one differentiable Newton step. This is mathematically
correct and is what makes the learned `c` trainable.

### Illustrative diff

Original (in-place, gradient-hostile):
```python
self.Xrekf = self.Xrekf.clone()
self.Xrekf[:, i + 1] = torch.squeeze(self.model.f(self.Xn[:, i]))
...
self.c_array.append(self.c.item())          # detached
self.th[i] = self.fnComputeTheta(P)         # non-differentiable in c
```

Modified (local state, autograd-safe, differentiable θ):
```python
x_seq.append(x_next)                        # build a list, stack at the end
...
self.c, h_t = self.nn(input_features, h_t)  # tensor, gradient kept
self.c_array.append(self.c)
th_i = self.fnComputeTheta(P)               # bisection + implicit-Newton dθ/dc
```

### Caveats (⚠️)

- **`fnComputeTheta` fallback branch** — `if not (g_low<=0 and g_high>=0):
  theta_hat = 0.5*high` is **independent of `c`**; it can fire silently when `P`
  is small, making `θ` insensitive to `c` in that regime. Latent issue.
- **`fnREKF` no longer returns `V`** (returns `None` in its place). Any caller
  relying on the covariance sequence from the return value would break; the
  original returned `self.V`.

---

## 3. `main_Robust_KNet.ipynb` — the driver (original 35 cells → current 51)

| Aspect | Original | Modified | Verdict |
|---|---|---|---|
| RT-KNet training | **single trajectory**/epoch, loss on **predicted** `Xrekf[:,:-1]`, `backward(retain_graph=True)` under `set_detect_anomaly(True)` | **mini-batch** (`n_batch` averaged, one step/epoch), **fixed CV set**, **posterior** loss, grad-norm diagnostics, `fcl_out` excluded from weight-decay, gradient clipping | ✅ mini-batch BPTT |
| Construction | `hidden_layers=[20,20,20,20,20]` | `gru_hidden_size=64` | ✅ matches GRU |
| REKF eval | predicted MSE only | + filtered MSE (`Xn_out`) for fairness | ✅ |
| New sections | — | Option-B partial-information experiment; `c_t`/θ/constant-`c`-sweep/solver-coherence diagnostics | ✅ useful |
| Performance | — | pre-allocated results dataframe (`O(n²)→O(n)`) | ✅ real fix |

**Note on the original training cell:** the `retain_graph=True` +
`set_detect_anomaly(True)` combination is a tell-tale sign the original authors
were fighting exactly the autograd/in-place problem that Task 1 required fixing —
which our local-state + differentiable-θ rewrite resolves.

**Note on the import trap:** the original notebook imports
`from RobustKalmanPY.robust_kalman import RobustKalman`. Since `robust_kalman.py`
is now the *modified* file, running `main_Robust_KNet_original.ipynb` today
silently uses the modified filter, **not** the original. To run the true original
("existing robust KalmanNet") we added frozen copies:
`KNet/RT_KalmanNet_nn_original.py` + `RobustKalmanPY/robust_kalman_original.py`.

---

## 4. Overall assessment

**Done correctly**
- ✅ Task 1 (BPTT): network (GRU hidden state), filter (local-state + `stack`,
  gradient-carrying `c`, implicit-Newton θ), and training loop (mini-batch,
  fixed CV, posterior loss) are all correct and mutually consistent.
- ✅ Task 2 (GRU): dense DNN-with-feedback → MLP encoder + GRU + head.
- ✅ `c_t`-collapse fixes: feature scaling, bias init, weight-decay exclusion on
  the output head — all well-motivated and effective.
- ✅ Fairness/robustness plumbing: posterior `Xn_out`, filtered-MSE reporting,
  the O(n²) dataframe fix.

**To flag / fix**
- ❌/⚠️ **`c_range=0.2` ceiling** — a regression vs the original's reach to ≈1;
  the primary reason `c` shows no leverage under mismatch and can't hit Task-3's
  `c≈1`. Recommend widening to ≈1.0–2.0.
- ⚠️ **`fnComputeTheta` fallback branch** is `c`-independent — replace the crude
  bracket-validity fallback with a proper residual/value check.
- ⚠️ **`fnREKF` return signature** dropped `V` — restore it or update all callers
  deliberately.

**Not a bug we introduced**
- The original network's inability to learn `c` (`grad=None`) is the *starting
  condition*, not a regression — it is exactly what Tasks 1–2 were meant to fix.

---

## 5. Recommended follow-ups (in priority order)

1. Widen `c_range` (net) to ≈1.0–2.0 and re-check the constant-`c` sweep — the
   single highest-leverage change.
2. Replace the `fnComputeTheta` fallback branch with a residual-based check.
3. Decide on `fnREKF`'s `V` return (restore, or update callers).
4. Keep the frozen `*_original.py` pair as the "existing robust KalmanNet"
   baseline for the Task-4 comparison.
