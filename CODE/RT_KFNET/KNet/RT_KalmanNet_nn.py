# -*- coding: utf-8 -*-
"""
In this file we build the neural network that estimates the robustness
parameter c of the REKF.
It is composed by
1) a fully connected encoder (Linear -> LayerNorm -> GELU -> Linear -> GELU)
2) a single GRU layer, whose hidden state carries the recurrent memory
3) a fully connected output layer followed by a sigmoid

Task 1 (BPTT): the recurrent hidden state h_t is never detached inside
forward(); it is an explicit input/output of forward() and its lifecycle
(when to start fresh, when to cut the graph for truncated BPTT) is entirely
owned by the caller (see RobustKalmanPY.robust_kalman.RobustKalman.fnREKF),
exactly like the filter's own recursive state (Xrekf_prev, Xn_prev, V_prev).
This keeps every "cut the graph here" decision in one place, instead of
splitting it between a reset_hidden() method on the network and detach()
calls on the filter state.

Task 2 (GRU architecture): the hand-written feedback loop of the original
implementation (a plain Linear layer fed with its own previous output) is
replaced by a proper nn.GRU cell, i.e. the ARCH1 architecture of "KalmanNet:
Neural Network Aided Kalman Filtering for Partially Known Dynamics" (FC input
layer -> GRU -> FC output layer), adapted to a scalar output.
"""

import torch
import torch.nn as nn


class RT_KalmanNet_nn(nn.Module):
    def __init__(self, input_size_fcl, gru_hidden_size=64, gru_layers=1, dropout=0.0):
        """
        Args:
            input_size_fcl (int): Numero di neuroni in ingresso all'encoder.
            gru_hidden_size (int): Dimensione dello stato nascosto della GRU
                (= dimensione dell'output dell'encoder).
            gru_layers (int): Numero di layer della GRU.
            dropout (float): Dropout tra i layer della GRU (ignorato se gru_layers == 1).
        """
        super().__init__()

        self.gru_hidden_size = gru_hidden_size
        self.gru_layers = gru_layers

        # Fully connected encoder: sostituisce il singolo FC layer d'ingresso
        # dell'originale con due layer (LayerNorm + GELU) per una rappresentazione
        # più espressiva dell'input, mantenendo la stessa dimensione di uscita
        # (= input della GRU).
        self.enc = nn.Sequential(
            nn.Linear(input_size_fcl, gru_hidden_size),
            nn.LayerNorm(gru_hidden_size),
            nn.GELU(),
            nn.Linear(gru_hidden_size, gru_hidden_size),
            nn.GELU(),
        )

        # Single GRU block: replaces the hand-written feedback loop.
        # batch_first=False (default): input/output shape [seq_len, batch, features],
        # coerente con l'uso a singolo timestep (seq_len=batch=1) e con la
        # convenzione di KalmanNet_TSP (ARCH1).
        self.gru = nn.GRU(
            input_size=gru_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            dropout=(dropout if gru_layers > 1 else 0.0),
            batch_first=False,
        )

        # Fully connected output layer
        self.output_layer = nn.Linear(gru_hidden_size, 1)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def init_hidden(self, batch_size: int = 1, device=None, dtype=None):
        """
        Costruisce uno stato nascosto iniziale nullo, shape [gru_layers, batch_size,
        gru_hidden_size]. Da chiamare esplicitamente dal chiamante a inizio
        sequenza (e passare poi h_t di ritorno in forward ad ogni step successivo).
        """
        return torch.zeros(self.gru_layers, batch_size, self.gru_hidden_size, device=device, dtype=dtype)

    def forward(self, x, h_t):
        """
        Args:
            x (Tensor): input della rete (vettore riga di dimensione input_size_fcl,
                relativo a un singolo timestep).
            h_t (Tensor): stato nascosto della GRU al passo precedente, shape
                [gru_layers, batch_size, gru_hidden_size]. Non opzionale: il
                chiamante deve fornirlo esplicitamente (vedi init_hidden()) cosi'
                che sia sempre chiaro, nel punto in cui si fa BPTT/TBPTT, quale
                sia la variabile che porta il gradiente da uno step al successivo.

        Returns:
            c_t (Tensor): scalare in (0, 1), il parametro di robustezza stimato.
            h_new (Tensor): nuovo stato nascosto della GRU, da ripassare alla
                chiamata successiva di forward().
        """
        z = self.enc(x)

        # nn.GRU (batch_first=False) expects input of shape [seq_len, batch, features].
        z = z.view(1, 1, -1)

        # h_t non viene mai detached qui dentro: il gradiente puo' fluire
        # all'indietro nel tempo attraverso h_t finche' il chiamante non lo
        # tronca esplicitamente (h_t.detach()) per il BPTT troncato.
        out, h_new = self.gru(z, h_t)

        raw = self.output_layer(out.view(1, -1))
        c_t = torch.sigmoid(raw).squeeze()

        return c_t, h_new