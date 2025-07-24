#! ~/anaconda3/envs/alpha/bin/python

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

import pandas as pd
import pandas_ta as ta

# pandas performance warning
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def suppress_consecutive_repeats(arr):
    arr = np.asarray(arr)
    out = np.zeros_like(arr)
    prev = None

    for i in range(len(arr)):
        if arr[i] != 0:
            if arr[i] != prev:
                out[i] = arr[i]
        prev = arr[i]

    return out


def calculate_sharpe(my_ret: np.ndarray):

    if np.all(np.isnan(my_ret)):
        return np.nan

    std = np.nanstd(my_ret, ddof=1)
    mean = np.nanmean(my_ret)

    if std == 0:
        return np.inf
    return mean / std


def calculate_pr(my_ret: np.ndarray):

    pos_sum = np.nansum(my_ret[my_ret > 0])
    neg_sum = np.nansum(my_ret[my_ret < 0])

    if np.isnan(pos_sum) or np.isnan(neg_sum):
        profit_ratio = np.nan
    elif neg_sum == 0:
        profit_ratio = np.inf if pos_sum > 0 else np.nan  # Avoid divide-by-zero
    else:
        profit_ratio = pos_sum / abs(neg_sum)
    return profit_ratio


def compute_returns(
    signal: pd.Series,
    log_ret: pd.Series,
    cost: float = 0.00,
):

    position_change = signal.diff().abs().fillna(0)
    position_change = np.array(position_change.values)

    next_r = log_ret.shift(-1)
    next_r = np.array(next_r.values)

    signal = np.array(signal.values)

    if len(signal) == 0 or len(next_r) == 0:
        return np.full_like(next_r, np.nan), np.nan, np.nan

    # Strategy return net of transaction costs
    my_ret = next_r * signal - position_change * cost

    sharpe = calculate_sharpe(my_ret)
    profit_ratio = calculate_pr(my_ret)

    return my_ret, profit_ratio, sharpe


def compute_returns_nn(
    df_trading: pd.DataFrame, cost: float = 0.001, thr: float = 0.005
):

    signal = df_trading["raw_signal"]
    pct_change = df_trading["close"].pct_change()
    next_r = df_trading["log_ret"].shift(-1)
    trade_signal = np.zeros_like(signal)
    trade_signal = np.zeros_like(signal)
    in_trade = False
    entry_index = None

    for i in range(1, len(signal)):
        if not in_trade and signal[i] == 1 and signal[i - 1] == 0:
            # Start trade
            in_trade = True
            trade_signal[i] = 1
        elif in_trade:
            if signal[i] == 0:
                # End of signal block
                in_trade = False
            elif pct_change[i] >= thr:
                # Exit due to threshold breach
                trade_signal[i] = 1
                in_trade = False
            else:
                # Continue trade
                trade_signal[i] = 1

    trade_signal = pd.DataFrame({"trade_signal": trade_signal})
    position_change = (
        signal.diff().abs().fillna(0)
    )  # First row has no previous position

    my_ret = next_r * signal - position_change * cost

    pr = np.nansum(my_ret[my_ret > 0]) / np.abs(np.nansum(my_ret[my_ret < 0]))
    sharpe = np.nanmean(my_ret.dropna()) / np.nanstd(my_ret.dropna(), ddof=1)

    return my_ret, pr, sharpe, trade_signal


def compute_win_ratio(signal, log_return, horizon=4) -> Tuple[float, np.ndarray]:
    trades = []

    i = 0
    while i < len(signal):
        s = signal[i]
        if s == 1 or s == -1:
            trade_return = s * np.sum(
                log_return[i + 1 : i + 1 + horizon]
            )  # skip signal bar itself
            trades.append(trade_return)
            i += horizon  # Skip to end of trade
        else:
            i += 1

    trades = np.array(trades)
    win_ratio = np.mean(trades > 0) if len(trades) > 0 else np.nan
    return win_ratio, trades


def generate_positions(signal, horizon=4) -> np.ndarray:
    signal = np.asarray(signal)
    positions = np.zeros_like(signal)
    i = 0
    while i < len(signal):
        if signal[i] == 1:  # long entry
            positions[i : i + horizon] = 1
            i += horizon  # skip ahead, no overlapping positions
        elif signal[i] == -1:  # short entry
            positions[i : i + horizon] = -1
            i += horizon
        else:
            i += 1
    return positions


def add_mas(
    df: pd.DataFrame, windows: List[int] | np.ndarray = np.arange(2, 25)
) -> pd.DataFrame:

    for window_length in windows:
        df[f"means_{window_length}"] = df["close"].rolling(window_length).mean()
        df[f"std{window_length}"] = df["close"].rolling(window_length).std(ddof=1)

    return df


def add_rsis(df: pd.DataFrame, windows: List[int] | np.ndarray = np.arange(2, 25)):

    for window_length in windows:
        df[f"rsi_{window_length}"] = ta.rsi(df["close"], length=window_length)

    return df


def add_vols(df: pd.DataFrame, windows: List[int] | np.ndarray = np.arange(2, 25)):

    if "log_ret" not in list(df.columns):
        df["log_ret"] = np.log(df["close"]).diff()

    for window_length in windows:
        df[f"vol_{window_length}"] = df["log_ret"].rolling(window_length).std(ddof=1)

    return df


# def add_rsis(df: pd.DataFrame, windows: List[int] | np.ndarray = np.arange(2, 25)):

#     if "log_ret" not in df.columns:
#         df["log_ret"] = np.log(df["close"]).diff()

#     for window_length in windows:
#         df[f"rsi_{window_length}"] = ta.rsi(df["close"], length=window_length)

#     return df


def get_features_names(df: pd.DataFrame) -> List[str]:

    feature_names = []
    for col in list(df.columns):

        if "mean" in col:
            feature_names.append(col)
        if "std" in col:
            feature_names.append(col)
        if "rsi" in col:
            feature_names.append(col)

    return feature_names


def get_features_matrix(df: pd.DataFrame) -> pd.DataFrame:

    feature_names = get_features_names(df)

    df_f = df[feature_names]

    return df_f
