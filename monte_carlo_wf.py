#! ~/anaconda3/bin/python

import numpy as np
import pandas as pd
import utils
import learn
from typing import Tuple, List


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
    df_trading: pd.DataFrame,
    train_period: pd.DateOffset,
    test_period: pd.DateOffset,
    n_samples: int = 500,
    block_size: int = 20,
):

    p0 = df_trading["close"][0]

    surrogates = [
        generate_bootstrap_series(df_trading["log_ret"], block_size, seed=i)
        for i in range(n_samples)
    ]

    profit_ratios = []
    sharpe_ratios = []
    sim_return = []

    for i, sample in enumerate(surrogates):

        sample.columns = ["log_ret"]
        sample["time"] = df_trading["time"]

        sample["close"] = p0 * np.exp(np.cumsum((sample["log_ret"])))
        sample = utils.add_rsis(sample)

        _, _, trading_sim = learn.walk_forward(
            sample, train_period, test_period, N=2, qmin=0.025, qmax=0.975
        )

        _my_ret, _pr, _sharpe = compute_returns(trading_sim)

        sim_return.append(_my_ret)
        profit_ratios.append(_pr)
        sharpe_ratios.append(_sharpe)

        if i % 50 == 0:
            print(f"Simulation num: {i}, profit ratio: {_pr},  sharpe ratio: {_sharpe}")

    return sim_return, profit_ratios, sharpe_ratios
