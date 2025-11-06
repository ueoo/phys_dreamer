import torch
import torch.nn as nn

from utils.dct import dct, dct3d, idct, idct_3d


class DCTTrajctoryField(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        pass

    def forward(self, x):
        pass

    def query_points_at_time(self, x, t):
        pass
