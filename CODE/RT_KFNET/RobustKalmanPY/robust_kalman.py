#%% Basic libraries for matrix manipulation and math functions
#
# v3: combines the two independent implementations of Task 1 (BPTT) + Task 2
# (GRU), built against KNet.RT_KalmanNet_nn_v3 (see ../modifiche_per_task12.md).
# Structure kept close to robust_kalman_original.py (same method names, same
# per-step operation order, same A/C/G/th bookkeeping); the state-management
# fix needed for multi-epoch BPTT uses local-variable-per-step + stack instead
# of preallocated-buffer + clone (see docs/team comparison for the rationale:
# the clone pattern is O(T^2) in time/memory because it copies the whole
# buffer at every step, while list+stack is O(T)).

import torch
import time
import contextlib
import torch.nn.functional as func
from KNet.RT_KalmanNet_nn_v3 import RT_KalmanNet_nn

#%%
# NOTE! There is a combination of numpy and torch thus if changing something use Tensors!
class RobustKalman():
    def __init__(self, SysModel, test_data, c : float = 1e-3, hard_coded: bool = False, use_nn: bool = False, input_feat_mode: int = 0, gru_hidden_size: int = 64, gru_layers: int = 1, dropout: float = 0.0, sl_model: int = 0, set_noise_matrices: bool = False, Q_mat = torch.eye(3), R_mat = torch.eye(3)):

        # Select whether to use the NN or regular REKF model
        self.use_nn = use_nn

        # Import the model from the SysModel class
        self.model = SysModel
        self.x0 = torch.transpose(SysModel.m1x_0, 0, 1)
        # Setting the noise covariance matrices
        if set_noise_matrices:
            # In case the sys_model matrices are not used
            self.Q = Q_mat
            self.R = R_mat
        else:
            # In case we are using the same noise matrices which generated the data
            self.Q = SysModel.Q
            self.R = SysModel.R
        self.T = SysModel.T
        self.y = test_data
        self.c = torch.tensor(c)
        self.hard_coded = hard_coded
        self.sl_model = sl_model

        # Preallocation of memory for the computation
        self.n = torch.Tensor.numpy(self.Q).shape[0] #state dimension
        self.p = torch.Tensor.numpy(self.R).shape[0] #output dimension

        if self.use_nn:
            self.Xrekf = torch.zeros(self.n, self.T+1, requires_grad=True)
        else:
            self.Xrekf = torch.zeros(self.n, self.T+1)

        self.Xn = torch.zeros(self.n, self.T)
        self.V = torch.zeros(self.n, self.n, self.T+1)

        self.reset_state()  # sets Xrekf_prev, y_prev, Xn_prev, V_prev

        self.A = torch.zeros(self.n, self.n, self.T)
        self.C = torch.zeros(self.p, self.n, self.T)
        self.G = torch.zeros(self.n, self.p, self.T)
        self.th = torch.zeros(self.T)
        self.theta_residual_array = torch.zeros(self.T)

        if self.use_nn:
            self.input_feat_mode = input_feat_mode
            # Select the input feature set
            if self.input_feat_mode == 0:   # Case {F2}
                input_size_fcl = self.p
            elif self.input_feat_mode == 1 or self.input_feat_mode == 2:    # Case {F1,F2,F4}, {F1,F3,F4}
                input_size_fcl = self.p + self.p + self.n
            elif self.input_feat_mode == 3:     # Case {F1,F2,F3,F4}
                input_size_fcl = self.p + self.p + self.n + self.n
            else:
                raise SystemExit("'input_feat_mode' must be an integer value between 0 and 3")

            # Initialize NN
            self.nn = RT_KalmanNet_nn(
                input_size_fcl=input_size_fcl,
                gru_hidden_size=gru_hidden_size,
                gru_layers=gru_layers,
                dropout=dropout,
            )

    def reset_state(self):
        self.Xrekf_prev = self.x0.squeeze(0).detach().clone()
        self.y_prev = torch.zeros(self.p, device=self.Xrekf_prev.device, dtype=self.Xrekf_prev.dtype)
        self.Xn_prev = torch.zeros(self.n, device=self.Xrekf_prev.device, dtype=self.Xrekf_prev.dtype)
        self.V_prev = (1e-3 * torch.eye(self.n, device=self.Xrekf_prev.device,
                                         dtype=self.Xrekf_prev.dtype)).detach()

    # Below one can choose to use either the closed form Jacobian or the numerical one from Pytorch
    def fnComputeJacobianF(self, x_n_temp):
        if self.hard_coded:
            f_jac = torch.tensor([[self.model.alpha*self.model.beta*(torch.cos(self.model.phi+ self.model.beta*x_n_temp[0])), 0],[0, self.model.alpha*self.model.beta*(torch.cos(self.model.phi+ self.model.beta*x_n_temp[1]))]])
        else:
            f_jac = torch.autograd.functional.jacobian(self.model.f, x_n_temp)
        return f_jac

    def fnComputeJacobianH(self, x_rekf_temp):
        if self.hard_coded:
            h_jac = torch.tensor([[2*self.model.a*self.model.b*(self.model.c + self.model.b*x_rekf_temp[0]), 0],[0, 2*self.model.a*self.model.b*(self.model.c + self.model.b*x_rekf_temp[1])]])
        else:
            h_jac = torch.autograd.functional.jacobian(self.model.h,x_rekf_temp)

        return h_jac

    def fnGamma(self, P_pred, theta):
        """gamma(P, theta) = tr[(I - theta*P)^-1 - I] + log det(I - theta*P), see [UAVF] eq. for theta_k."""
        I = torch.eye(self.n)
        M = I - theta * P_pred
        return torch.trace(torch.linalg.solve(M, I) - I) + torch.log(torch.det(M))

    def fnComputeTheta(self, P_pred):
        # The root theta* of gamma(P, theta) = c is found by bisection. This search
        # is an inherently non-differentiable, iterative procedure (the update at
        # each step is chosen by an if-branch on a bisection residual, not by an
        # arithmetic expression in c), so gradients w.r.t. self.c would NOT flow
        # through it: torch.no_grad() makes this explicit and avoids building a
        # graph for the (many) intermediate iterations.
        with torch.no_grad():
            value = torch.tensor([1.0])
            t1 = torch.tensor([0.0])
            e = torch.linalg.eig(P_pred)[0]
            r = torch.max(torch.abs(e))
            t2 = (1 - 1e-5) * (torch.pow(r, -1))
            c_value = self.c.detach() if torch.is_tensor(self.c) else self.c

            while torch.abs(value) >= 1e-5:
                theta_star = 0.5 * (t1 + t2)
                value = self.fnGamma(P_pred, theta_star) - c_value
                if value > 0:
                    t2 = theta_star
                else:
                    t1 = theta_star

        with torch.no_grad():
            self._last_theta_residual = torch.abs(self.fnGamma(P_pred, theta_star) - c_value)

        # theta_star is the numerically converged root, but it carries no gradient
        # w.r.t. self.c (it was computed entirely under no_grad). The REKF baseline
        # (use_nn=False, fixed scalar c) never backprops through theta, so skip the
        # extra autograd.grad() work below and return the bisection result directly.
        if not self.use_nn:
            return theta_star

        # theta_star is the numerically converged root, but it carries no gradient
        # w.r.t. self.c (it was computed entirely under no_grad). To let gradients
        # flow from theta back to the NN weights (via self.c), we take a single
        # differentiable implicit-function-theorem correction step around theta_star:
        # since gamma(P, theta(c)) = c identically, d theta/d c = 1 / gamma'(P, theta).
        # This reconstructs (to first order, i.e. exactly at convergence since
        # gamma(P, theta_star) ~= c) the same numeric value while making theta a
        # differentiable function of self.c. torch.enable_grad() makes this correction
        # step work even when fnComputeTheta is itself called inside an outer
        # torch.no_grad() block (eval mode in fnREKF), without forcing the caller to
        # track gradients through the (expensive, non-differentiable-anyway) bisection.
        with torch.enable_grad():
            theta_star = theta_star.detach().clone().requires_grad_(True)
            gamma_at_star = self.fnGamma(P_pred, theta_star)
            # gamma_grad = d gamma / d theta at theta_star is treated as a constant
            # scaling factor (create_graph=False): we only need one derivative here,
            # not a doubly-differentiable graph through the correction itself.
            gamma_grad = torch.autograd.grad(gamma_at_star, theta_star, create_graph=False)[0]

            c_for_grad = self.c if torch.is_tensor(self.c) else torch.tensor(float(self.c))
            theta = theta_star.detach() + (c_for_grad - gamma_at_star.detach()) / gamma_grad

        return theta

    # Computation of the REKF
    def fnREKF(self, train: bool = False, reset: bool = True, bptt_truncation: int = None):
        """
        Args:
            train (bool): if True, the NN (when use_nn) is set to train() mode and the
                whole forward pass builds a computational graph so that BPTT can be
                applied by the caller via loss.backward(). If False, the NN is set to
                eval() mode and the forward pass runs under torch.no_grad() (no graph
                is built at all).
            reset (bool): if True (default), the filter's recursive state
                (Xrekf_prev, Xn_prev, V_prev, y_prev) is reset to its initial value
                before running. Set to False to continue from where the previous
                call to fnREKF left off (e.g. for TBPTT across successive calls).
            bptt_truncation (int, optional): if None (default), the graph is left
                untouched across all T time steps -> full BPTT. If set to a positive
                integer K, every K time steps the recurrent state of the NN (h_t) and
                the filter's own recursive state (Xrekf_prev, Xn_prev, V_prev) are
                detached from the graph, implementing truncated BPTT (TBPTT) as
                discussed in [TSP] Sec. "Training Algorithm". Ignored when use_nn is
                False.
        """
        if reset:
            self.reset_state()

        if self.use_nn:
            if train:
                self.nn.train()
            else:
                self.nn.eval()
            self.c_array = []

        # Local recursive state: starts detached from whatever graph the previous
        # call to fnREKF (and the .backward() + optimizer.step() possibly run in
        # between) left attached to self.Xrekf_prev/Xn_prev/V_prev/y_prev. Without
        # this, a second call would keep referencing NN weights that the optimizer
        # has since mutated in-place, raising a version-mismatch RuntimeError on
        # backward.
        x_prev = self.Xrekf_prev.detach().clone()
        y_prev = self.y_prev.detach().clone()
        xn_prev = self.Xn_prev.detach().clone()
        V_prev = self.V_prev.detach().clone()

        # RT_KalmanNet_nn_v3 requires an explicit h_t (not optional): built here,
        # at the start of the sequence, exactly where the filter's own state is
        # reset above, so that "wiping the NN's memory" and "wiping the filter's
        # memory" stay conceptually aligned.
        h_t = self.nn.init_hidden(device=x_prev.device, dtype=x_prev.dtype) if self.use_nn else None

        x_seq = [x_prev]     # builds Xrekf without in-place writes on a persistent buffer
        xn_seq = []          # filtered/posterior estimates Xn_i (use y_i)
        A_seq = []
        C_seq = []
        G_seq = []
        th_seq = []
        theta_residual_seq = []

        start = time.time()

        # In eval mode (train=False) we never need gradients: wrap the whole loop in
        # torch.no_grad() to avoid building a computational graph. In train mode (or
        # when the NN is not used at all) we keep autograd enabled as usual.
        no_grad_ctx = torch.no_grad() if (self.use_nn and not train) else contextlib.nullcontext()

        with no_grad_ctx:
            # Forward Step
            for i in range(0, self.T):

                # C_t
                C_i = self.fnComputeJacobianH(x_prev)

                # L_t
                L = V_prev @ torch.transpose(C_i, 0, 1) @ torch.linalg.solve(
                    C_i @ V_prev @ torch.transpose(C_i, 0, 1) + self.R,
                    torch.eye(self.p, device=V_prev.device, dtype=V_prev.dtype))

                # h(\hat x_t)
                hn = self.model.h(x_prev)

                # \hat x_t|t
                Xn_i = x_prev + (L @ (self.y[:, i] - hn))
                xn_seq.append(Xn_i)

                # A_t
                A_i = self.fnComputeJacobianF(Xn_i)

                # G_t
                G_i = A_i @ L

                # \hat x_t+1
                x_next = torch.squeeze(self.model.f(Xn_i))

                # P_t+1
                P = (A_i @ V_prev @ torch.transpose(A_i, 0, 1)
                     - A_i @ V_prev @ torch.transpose(C_i, 0, 1)
                     @ torch.linalg.solve(
                         C_i @ V_prev @ torch.transpose(C_i, 0, 1) + self.R,
                         torch.eye(self.p, device=V_prev.device, dtype=V_prev.dtype))
                     @ C_i @ V_prev @ torch.transpose(A_i, 0, 1)
                     + self.Q)

                if self.use_nn:
                    # Compute input features F1,F2,F3,F4 (no normalization: kept
                    # identical to the professor's original feature definitions
                    # until a controlled experiment shows rescaling helps).
                    f1 = self.y[:, i] - y_prev
                    f2 = self.y[:, i] - hn
                    f3 = Xn_i - xn_prev
                    f4 = Xn_i - x_prev

                    # Stacking input features [F1, F2, F4]
                    if self.input_feat_mode == 0:
                        input_features = f2
                    elif self.input_feat_mode == 1:
                        input_features = torch.cat([f1, f2, f4], dim=0)
                    elif self.input_feat_mode == 2:
                        input_features = torch.cat([f1, f3, f4], dim=0)
                    else:
                        input_features = torch.cat([f1, f2, f3, f4], dim=0)

                    # Forward Step
                    self.c, h_t = self.nn(input_features, h_t)
                    self.c_array.append(self.c)

                # th_t
                th_i = self.fnComputeTheta(P)

                # V_t+1
                V_i = torch.linalg.solve(
                    torch.linalg.solve(P, torch.eye(self.n, device=P.device, dtype=P.dtype))
                    - th_i * torch.eye(self.n, device=P.device, dtype=P.dtype),
                    torch.eye(self.n, device=P.device, dtype=P.dtype))

                x_seq.append(x_next)
                A_seq.append(A_i.detach())
                C_seq.append(C_i.detach())
                G_seq.append(G_i.detach())
                th_seq.append(th_i.detach())
                theta_residual_seq.append(self._last_theta_residual)

                # Update recursive state
                x_prev = x_next
                xn_prev = Xn_i
                y_prev = self.y[:, i]
                V_prev = V_i

                # Truncated BPTT: cut the graph every bptt_truncation steps, both for
                # the NN's own recurrent state and for the filter's recursive state.
                # h_t is detached (not reset to zero) so the recurrence itself keeps
                # running across the boundary -- only the gradient path is cut.
                if self.use_nn and train and bptt_truncation is not None and (i + 1) % bptt_truncation == 0:
                    h_t = h_t.detach()
                    x_prev = x_prev.detach()
                    xn_prev = xn_prev.detach()
                    V_prev = V_prev.detach()

        # Assemble outputs. Xrekf_out is the one-step-ahead PREDICTED sequence
        # (needs Xrekf_out[:, :-1] to compare against test_target); Xn_out is the
        # filtered/posterior sequence x_{i|i} (uses y_i), directly comparable to
        # test_target with no truncation, like KalmanNet's posterior output.
        Xrekf_out = torch.stack(x_seq, dim=1)
        self.Xn_out = torch.stack(xn_seq, dim=1)
        self.A = torch.stack(A_seq, dim=-1)
        self.C = torch.stack(C_seq, dim=-1)
        self.G = torch.stack(G_seq, dim=-1)
        self.th = torch.stack(th_seq)
        self.theta_residual_array = torch.stack(theta_residual_seq)
        # self.V is intentionally NOT kept as a stacked buffer here: V_i is a
        # small per-step covariance only needed one step ahead (V_prev), so
        # unlike A/C/G/th (diagnostics explicitly requested for inspection) it
        # is not accumulated. Callers needing V_i for a specific i can recompute
        # it from A/C/G/th if needed.

        # Persist state, detached, for the next call
        self.Xrekf_prev = x_prev.detach()
        self.Xn_prev = xn_prev.detach()
        self.y_prev = y_prev.detach()
        self.V_prev = V_prev.detach()

        end = time.time()
        t = end - start

        if self.use_nn:
            return [Xrekf_out, self.c_array, None, t]
        else:
            return [Xrekf_out, None, t]
