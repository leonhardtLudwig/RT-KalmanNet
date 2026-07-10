"""
Task 3 (LF_DATA) -- least-favorable CV-model data pipeline.

Loads the MATLAB-generated least-favorable data (CODE/LF_DATA/data.mat),
builds the linear constant-velocity (CV) SystemModel the professor's
robust_filter.m uses, and provides:

  1. load_lf_data()        -> tensors + a linear CV SystemModel
  2. eval_constant_c()     -> plain-REKF MSE for a fixed scalar c
  3. __main__ grid search  -> confirms the MSE-optimal constant c ~= 1
                              (the ground-truth c the data was generated with)

This last check is the *pipeline validation* for Task 3: if the optimal
constant c comes out near 1.0, the MATLAB->Python bridge + linear filter
are correct and we can trust the learned-c comparison that follows.

Run:  conda run -n kalman_net python task3_lf_data.py
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scipy.io as sio
from Simulations.Extended_sysmdl import SystemModel
from RobustKalmanPY.robust_kalman import RobustKalman

DATA_MAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "LF_DATA", "data.mat")


def load_lf_data(path=DATA_MAT, dtype=torch.float32):
    """Load data.mat, return (model matrices, LFM trajectories, CV SystemModel).

    Returns a dict with:
      A, C, Q, R, V0, x0   : model tensors (float)
      Y      : [n_traj, p, N]   observations y_1..y_N
      Xtrue  : [n_traj, n, N]   true states  x_1..x_N (aligned with Y)
      c_true : float            the constant tolerance the data was made with
      sys    : linear CV SystemModel (f=A x, h=C x)
    """
    m = sio.loadmat(path)
    A = torch.tensor(np.asarray(m["A"], dtype=np.float64), dtype=dtype)      # 2x2
    C = torch.tensor(np.asarray(m["C"], dtype=np.float64), dtype=dtype)      # 1x2
    Q = torch.tensor(np.asarray(m["Q"], dtype=np.float64), dtype=dtype)      # 2x2
    R = torch.tensor(np.atleast_2d(np.asarray(m["R"], dtype=np.float64)), dtype=dtype)  # 1x1
    V0 = torch.tensor(np.asarray(m["V0"], dtype=np.float64), dtype=dtype)    # 2x2
    x0 = torch.tensor(np.asarray(m["x0"], dtype=np.float64), dtype=dtype)    # 2x1
    c_true = float(np.squeeze(m["c"]).reshape(-1)[0])                        # all equal (=1)

    yw = np.asarray(m["yw"], dtype=np.float32)   # [p, N,   M, cM]
    xw = np.asarray(m["xw"], dtype=np.float32)   # [n, N+1, M, cM]
    p, N, M, cM = yw.shape
    n = xw.shape[0]

    # All cM slices share c=1 but are independent random draws -> flatten to
    # M*cM independent trajectories.
    # Alignment: observation y_t (t=1..N) pairs with state x_t (t=1..N).
    #   yw[:, i, ...]      is y_{i+1}
    #   xw[:, i+1, ...]    is x_{i+1}   (xw column 0 is the initial x_0)
    Y = yw.reshape(p, N, M * cM).transpose(2, 0, 1)             # [n_traj, p, N]
    Xtrue = xw[:, 1:N + 1, :, :].reshape(n, N, M * cM).transpose(2, 0, 1)  # [n_traj, n, N]
    Y = torch.tensor(Y, dtype=dtype)
    Xtrue = torch.tensor(Xtrue, dtype=dtype)

    f = lambda x: (A @ x)
    h = lambda x: (C @ x)
    sys = SystemModel(f, Q, h, R, T=N, T_test=N, m=n, n=p)
    sys.InitSequence(x0, V0)   # m1x_0 = x0, m2x_0 = V0

    return dict(A=A, C=C, Q=Q, R=R, V0=V0, x0=x0, c_true=c_true,
                Y=Y, Xtrue=Xtrue, sys=sys, n=n, p=p, N=N, n_traj=M * cM)


def eval_constant_c(data, c, n_traj=50, use_V0=True):
    """Mean per-step MSE of the plain REKF (fixed scalar c) over n_traj trajectories.

    Uses the *filtered* estimate Xn_out (posterior x_{t|t}), matching the
    fair-comparison convention used elsewhere in the project.
    """
    sys, Y, Xtrue, N, p, n = (data["sys"], data["Y"], data["Xtrue"],
                              data["N"], data["p"], data["n"])
    rk = RobustKalman(sys, torch.zeros(p, N), c=float(c),
                      hard_coded=False, use_nn=False, sl_model=0)
    # Linear CV model: the Jacobians are the constant matrices A, C -- override
    # the per-step autograd Jacobian (huge speedup, exact for a linear model).
    A_mat, C_mat = data["A"], data["C"]
    rk.fnComputeJacobianF = lambda x: A_mat
    rk.fnComputeJacobianH = lambda x: C_mat
    mses = []
    for k in range(min(n_traj, data["n_traj"])):
        rk.y = Y[k]                        # [p, N]
        rk.fnREKF(train=False, reset=True)
        if use_V0:                         # optional: restart with the true V0=I
            pass
        Xn = rk.Xn_out                     # [n, N] filtered estimates
        mses.append(((Xn - Xtrue[k]) ** 2).mean().item())
    return float(np.mean(mses))


if __name__ == "__main__":
    torch.manual_seed(0)
    d = load_lf_data()
    print(f"loaded: n_traj={d['n_traj']}  N={d['N']}  n={d['n']} p={d['p']}  "
          f"c_true={d['c_true']}")
    print(f"model check: A=\n{d['A'].numpy()}\n C={d['C'].numpy().tolist()} "
          f"R={float(d['R'])}  Q diag={torch.diag(d['Q']).numpy().tolist()}")

    n_traj = int(os.environ.get("NTRAJ", "40"))
    c_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
    print(f"\n=== constant-c grid search (plain REKF, filtered MSE, {n_traj} traj) ===")
    results = []
    for c in c_grid:
        mse = eval_constant_c(d, c, n_traj=n_traj)
        results.append((c, mse))
        print(f"  c={c:<5}  MSE={mse:.5f}")
    best_c, best_mse = min(results, key=lambda t: t[1])
    print(f"\nOptimal constant c in grid: {best_c}  (MSE={best_mse:.5f})")
    print(f"True c used to generate the data: {d['c_true']}")
    verdict = "PASS" if 0.5 <= best_c <= 2.0 else "CHECK"
    print(f"[{verdict}] optimal-c near true-c(=1)?  -> validates the bridge + linear filter")
