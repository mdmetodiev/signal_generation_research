#! ~/anaconda3/envs/alpha/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import learn
from typing import Tuple, Any

import numpy as np


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


from typing import Union


def permute_close_prices(
    close: Union[pd.Series, pd.DataFrame], start_index: int = 0, seed=None
) -> pd.Series:
    if isinstance(close, pd.DataFrame):
        close = close["close"]

    assert isinstance(close, pd.Series)
    assert start_index >= 0 and start_index < len(close)

    if seed is not None:
        rng = np.random.RandomState(seed)

    log_prices = np.log(close)
    log_returns = log_prices.diff().to_numpy()

    preserved = log_prices.iloc[: start_index + 1]

    to_permute = log_returns[start_index + 1 :]
    permuted_returns = np.random.permutation(to_permute)

    new_log_prices = list(preserved)
    for r in permuted_returns:
        new_log_prices.append(new_log_prices[-1] + r)

    new_log_prices = pd.Series(new_log_prices, index=close.index)
    return np.exp(new_log_prices)
