#! ~/anaconda3/bin/python


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import learn
from typing import Tuple, Any

from sklearn.decomposition import PCA

from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)



def create_blocks(series, block_size):
    n_blocks = len(series) - block_size + 1
    blocks = [series[i : i + block_size] for i in range(n_blocks)]
    return blocks


def generate_bootstrap_series(series, block_size, seed=None):
    if seed is not None:
        np.random.seed(seed)

    blocks = create_blocks(series, block_size)
    n_blocks = len(blocks)
    n_blocks_needed = int(np.ceil(len(series) / block_size))

    # Sample indices of blocks
    sampled_indices = np.random.choice(
        np.arange(n_blocks), size=n_blocks_needed, replace=True
    )
    sampled_blocks = [blocks[i] for i in sampled_indices]

    surrogate = np.concatenate(sampled_blocks)[: len(series)]

    surrogate = pd.DataFrame(surrogate, index=series.index)
    return surrogate


def compute_returns(
    df_trading: pd.DataFrame, cost: float = 0.001
):  # 0.1% transaction cost per unit change in position

    signal = df_trading["signal"]
    position_change = (
        signal.diff().abs().fillna(0)
    )  # First row has no previous position

    next_r = df_trading["log_ret"].shift(-1)

    # Strategy return net of transaction costs
    my_ret = next_r * signal - position_change * cost

    pr = np.nansum(my_ret[my_ret > 0]) / np.abs(np.nansum(my_ret[my_ret < 0]))
    sharpe = np.nanmean(my_ret.dropna()) / np.nanstd(my_ret.dropna(), ddof=1)

    return my_ret, pr, sharpe


def run_mc(
    df: pd.DataFrame,
    train_period: pd.DateOffset,
    test_period: pd.DateOffset,
    N:int,
    npca:int=2,
    n_samples: int = 500,
    block_size: int = 20,
):
    df.index = pd.to_datetime(df["time"].values)
    p0 = df["close"][0]


    profit_ratios = []
    sharpe_ratios = []
    sim_return = []

    for i in range(n_samples):
        pass
    