import torch
import torch.nn as nn
import torch.nn.functional as F

class RT_KalmanNet_nn(nn.Module):
    def __init__(
        self,
        input_size_fcl,
        gru_hidden_size=64,
        c_floor=1e-4,
        c_range=0.2,
        gru_layers=1,
        dropout=0.0
    ):
        super().__init__()

        self.c_floor = float(c_floor)
        self.c_range = float(c_range)

        self.enc = nn.Sequential(
            nn.Linear(input_size_fcl, gru_hidden_size),
            nn.LayerNorm(gru_hidden_size),
            nn.GELU(),
            nn.Linear(gru_hidden_size, gru_hidden_size),
            nn.GELU(),
        )

        self.gru = nn.GRU(
            input_size=gru_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            dropout=(dropout if gru_layers > 1 else 0.0),
            batch_first=True
        )

        self.fcl_out = nn.Linear(gru_hidden_size, 1)

        nn.init.xavier_uniform_(self.fcl_out.weight)
        nn.init.constant_(self.fcl_out.bias, 0.0)

    def forward(self, x, h_t=None):
        z = self.enc(x)

        if z.dim() == 1:
            z = z.unsqueeze(0).unsqueeze(0)
        elif z.dim() == 2:
            z = z.unsqueeze(1)

        out, h_new = self.gru(z, h_t)
        raw = self.fcl_out(out).squeeze()

        c_t = self.c_floor + self.c_range * torch.sigmoid(raw)
        return c_t, h_new