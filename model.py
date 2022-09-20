import numpy as np
import torch
import torch.nn as nn


class Q_Base_Net(nn.Module):
    def __init__(self, device='cuda', action_num=19):
        super().__init__()
        self.device = device
        self.hs, self.cs = None, None
        self.action_num = action_num

        self.init_layer = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(True)
        )

        # Flatten_size:2560

        self.lstm = nn.LSTMCell(2560, 256)

        self.value = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, action_num)
        )

    def forward(self, state, return_hs_cs=False):

        seq_size = 1
        batch_size, c, h, w = state.size()

        state = state.view(-1, 4, 72, 96)

        hs = self.init_layer(state).view(seq_size, batch_size, 2560)
        if self.hs is None:
            self.hs = torch.zeros(batch_size, 256).to(self.device)
            self.cs = torch.zeros(batch_size, 256).to(self.device)

        hs_seq = []
        cs_seq = []
        for h in hs:
            self.hs, self.cs = self.lstm(h, (self.hs, self.cs))

            hs_seq.append(self.hs)
            cs_seq.append(self.cs)
        hs_seq = torch.cat(hs_seq, dim=0)
        cs_seq = torch.cat(cs_seq, dim=0)

        val = self.value(hs_seq)
        adv = self.advantage(hs_seq)
        q_val = val + adv - adv.mean(1, keepdim=True)

        hs_seq = hs_seq.detach().cpu().numpy()
        cs_seq = cs_seq.detach().cpu().numpy()

        if return_hs_cs:
            return q_val, hs_seq, cs_seq
        else:
            return q_val

    def reset(self):
        self.hs, self.cs = None, None

    def set_state(self, hs, cs):
        self.hs, self.cs = hs, cs

    def get_state(self):
        return self.hs.detach().cpu().numpy(), self.cs.detach().cpu().numpy()
