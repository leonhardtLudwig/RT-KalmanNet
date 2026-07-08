#%% Basic libraries for matrix manipulation and math functions

import torch
import time
import torch.nn.functional as func
from KNet.RT_KalmanNet_nn import RT_KalmanNet_nn

# ==============================================================================
# TODO LIST: INTEGRAZIONE RETE-FILTRO (TASK 1 - Setup per BPTT in Loop Chiuso)
# ==============================================================================
#
# ✅ COSA FARE (Aggiungere/Modificare):
# [ ] Aggiornare l'istanziazione nel costruttore (__init__): Modificare la
#     chiamata a `RT_KalmanNet_nn` passando i nuovi parametri richiesti dalla
#     GRU (es. `gru_hidden_size` invece di `hidden_layers`).
# [ ] Inizializzare la memoria: Aggiungere `h_t = None` ESATTAMENTE PRIMA 
#     dell'inizio del ciclo `for i in range(0, self.T):` nella funzione `fnREKF`.
# [ ] Adattare le dimensioni dei tensori: Dentro il ciclo `for`, prima di passare 
#     `input_features` alla rete, assicurati che abbia forma 3D (Batch, Seq, Feat).
#     Usa `input_features = input_features.view(1, 1, -1)` o `.unsqueeze()`.
# [ ] Aggiornare la chiamata alla Rete: Sostituire `self.c = self.nn(input_features)`
#     con `self.c, h_t = self.nn(input_features, h_t)`.
#
# 🚫 COSA NON FARE (Lasciare intatto):
# [ ] NON modificare la matematica del Filtro: Tutte le equazioni (P, V, A, C, L)
#     e le Jacobiane sono corrette.
# [ ] NON modificare il calcolo delle feature (f1, f2, f3, f4).
# [ ] NON cercare di far girare l'intero vettore dei tempi in un colpo solo 
#     nella rete. A causa del loop chiuso (l'input t+1 dipende dall'uscita t),
#     devi per forza ciclare step-by-step aggiornando `h_t`.
#
# 🧠 NOTA CONCETTUALE:
# Srotolando `h_t` all'interno del ciclo `for`, stai costruendo un grafo
# computazionale continuo. Quando il ciclo finirà, PyTorch avrà tracciato
# l'intera storia e potrai chiamare `loss.backward()` sull'intera traiettoria.
# ==============================================================================

#%%
# NOTE! There is a combination of numpy and torch thus if changing something use Tensors!
class RobustKalman():
    def __init__(self, SysModel, test_data, c : float = 1e-3, hard_coded: bool = False, use_nn: bool = False, input_feat_mode: int = 0, gru_hidden_size: int = 64, sl_model: int = 0, set_noise_matrices: bool = False, Q_mat = torch.eye(3), R_mat = torch.eye(3), c_init: float = None):
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

        # Characteristic scales used to rescale the NN's input features (§2.4)
        self._obs_scale = torch.sqrt(torch.clamp(torch.diagonal(self.R), min=1e-6))
        self._state_scale = torch.sqrt(torch.clamp(torch.diagonal(self.Q), min=1e-6))

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

        self.reset_state()  # sets Xrekf_prev, y_prev, Xn_prev, V_prev (was inlined here before)

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

            resolved_c_init = float(c) if c_init is None else float(c_init)
            self.nn = RT_KalmanNet_nn(
                input_size_fcl=input_size_fcl,
                gru_hidden_size=gru_hidden_size,
                c_floor=1e-4,
                c_range=0.2,
                c_init=resolved_c_init,
                gru_layers=1,
                dropout=0.0
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
    
    def fnComputeTheta(self, P_pred):
        """
        Implicit-layer style solve for theta_t from:
            g(theta, c) = tr((I - theta P)^-1 - I) + logdet(I - theta P) - c = 0

        Forward: Solve root with bisection in no_grad for numerical robustness.
        Backward: Apply one differentiable Newton refinement around theta_hat.
        """
        eps = torch.tensor(1e-8, device=P_pred.device, dtype=P_pred.dtype)
        one = torch.tensor(1.0, device=P_pred.device, dtype=P_pred.dtype)

        # Symmetrize P for numerical stability
        P = 0.5 * (P_pred + P_pred.transpose(0, 1))
        I_n = torch.eye(self.n, device=P.device, dtype=P.dtype)

        # c_t from GRU
        if torch.is_tensor(self.c):
            c_val = self.c.to(device=P.device, dtype=P.dtype)
        else:
            c_val = torch.tensor(self.c, device=P.device, dtype=P.dtype)

        c_val = torch.clamp(c_val, min=eps)

        # Spectral upper bound
        eigvals = torch.linalg.eigvalsh(P)
        r = torch.clamp(torch.max(torch.abs(eigvals)), min=eps)
        theta_max = (one - 1e-5) / r
        theta_hi = theta_max * (one - 1e-6)

        def g_fn(theta_scalar, c_scalar):
            M = I_n - theta_scalar * P
            Minv = torch.linalg.solve(M, I_n)
            trace_term = torch.trace(Minv - I_n)
            sign, logabsdet = torch.linalg.slogdet(M)
            logdet_term = torch.where(sign > 0, logabsdet, torch.log(eps))
            return trace_term + logdet_term - c_scalar

        # ====================================================================
        # 1) Robust root-finding ALWAYS in no_grad (numerical stability)
        #    -- unchanged from before --
        # ====================================================================
        with torch.no_grad():
            low = torch.zeros((), device=P.device, dtype=P.dtype)
            high = theta_hi

            g_low = g_fn(low, c_val.detach())
            g_high = g_fn(high, c_val.detach())

            if not (g_low <= 0 and g_high >= 0):
                theta_hat = 0.5 * high
            else:
                for _ in range(50):
                    mid = 0.5 * (low + high)
                    g_mid = g_fn(mid, c_val.detach())
                    if g_mid > 0:
                        high = mid
                    else:
                        low = mid
                theta_hat = 0.5 * (low + high)

        # ====================================================================
        # 2) Differentiable Newton refinement -- ALWAYS run now, regardless of
        #    ambient no_grad() context, so the returned VALUE is identical in
        #    train and eval. torch.enable_grad() locally re-enables autograd
        #    even if the caller is inside torch.no_grad().
        # ====================================================================
        grad_was_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            theta = theta_hat.detach().clone()
            theta.requires_grad_(True)
            g_val = g_fn(theta, c_val)

            dg_dtheta = torch.autograd.grad(
                g_val, theta, create_graph=True, retain_graph=True
            )[0]

            dg_abs = torch.abs(dg_dtheta)
            dg_safe = torch.where(
                dg_abs > 1e-10,
                dg_dtheta,
                torch.sign(dg_dtheta) * torch.tensor(1e-10, device=P.device, dtype=P.dtype)
            )

            theta_refined = theta - g_val / dg_safe
            theta_refined = torch.clamp(theta_refined, min=torch.tensor(0.0, device=P.device, dtype=P.dtype), max=theta_hi)

        # Diagnostic: residual |g(theta,c)| at the refined root (near 0 = solver converged)
        with torch.no_grad():
            self._last_theta_residual = torch.abs(g_fn(theta_refined.detach(), c_val.detach()))

        # If the caller was in no_grad(), drop the small graph we just built
        # (pure efficiency cleanup -- the numeric value is the same either way)
        return theta_refined if grad_was_enabled else theta_refined.detach()


    def fnREKF(self, train: bool = False, reset: bool = True):
        if reset:
            self.reset_state()

        if self.use_nn:
            if train:
                self.nn.train()
            else:
                self.nn.eval()
            self.c_array = []

        start = time.time()

        # Stato locale: evita riferimenti al grafo della chiamata precedente
        x_prev = self.Xrekf_prev.detach().clone()
        y_prev = self.y_prev.detach().clone()
        xn_prev = self.Xn_prev.detach().clone()
        V_prev = self.V_prev.detach().clone()

        h_t = None
        x_seq = [x_prev]  # per costruire Xrekf senza in-place su buffer persistente

        for i in range(0, self.T):
            C_i = self.fnComputeJacobianH(x_prev)

            L = V_prev @ torch.transpose(C_i, 0, 1) @ torch.linalg.solve(
                C_i @ V_prev @ torch.transpose(C_i, 0, 1) + self.R,
                torch.eye(self.p, device=V_prev.device, dtype=V_prev.dtype),
            )

            hn = self.model.h(x_prev)
            Xn_i = x_prev + (L @ (self.y[:, i] - hn))

            A_i = self.fnComputeJacobianF(Xn_i)

            x_next = torch.squeeze(self.model.f(Xn_i))

            P = (
                A_i @ V_prev @ torch.transpose(A_i, 0, 1)
                - A_i @ V_prev @ torch.transpose(C_i, 0, 1)
                @ torch.linalg.solve(
                    C_i @ V_prev @ torch.transpose(C_i, 0, 1) + self.R,
                    torch.eye(self.p, device=V_prev.device, dtype=V_prev.dtype),
                )
                @ C_i @ V_prev @ torch.transpose(A_i, 0, 1)
                + self.Q
            )

            if self.use_nn:
                f1 = (self.y[:, i] - y_prev) / self._obs_scale
                f2 = (self.y[:, i] - hn) / self._obs_scale
                f3 = (Xn_i - xn_prev) / self._state_scale
                f4 = (Xn_i - x_prev) / self._state_scale

                if self.input_feat_mode == 0:
                    input_features = f2
                elif self.input_feat_mode == 1:
                    input_features = torch.cat([f1, f2, f4], dim=0)
                elif self.input_feat_mode == 2:
                    input_features = torch.cat([f1, f3, f4], dim=0)
                else:
                    input_features = torch.cat([f1, f2, f3, f4], dim=0)

                input_features = input_features.view(1, 1, -1)
                self.c, h_t = self.nn(input_features, h_t)
                self.c = self.c.squeeze()
                self.c_array.append(self.c)

            th_i = self.fnComputeTheta(P)
            self.th[i] = th_i.detach()
            self.theta_residual_array[i] = self._last_theta_residual

            V_i = torch.linalg.solve(
                torch.linalg.solve(
                    P, torch.eye(self.n, device=P.device, dtype=P.dtype)
                ) - th_i * torch.eye(self.n, device=P.device, dtype=P.dtype),
                torch.eye(self.n, device=P.device, dtype=P.dtype),
            )

            x_seq.append(x_next)

            # update stato locale
            x_prev = x_next
            xn_prev = Xn_i
            y_prev = self.y[:, i]
            V_prev = V_i

        Xrekf_out = torch.stack(x_seq, dim=1)

        # Persiste stato solo detached per la prossima chiamata
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