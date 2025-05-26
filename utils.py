#! ~/anaconda3/bin/python
import numpy as np
import matplotlib.pyplot as plt

from typing import List

import pandas as pd
import pandas_ta as ta


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

def compute_returns_nn(
    df_trading: pd.DataFrame, cost: float = 0.001, thr:float = 0.005
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

    trade_signal = pd.DataFrame({"trade_signal":trade_signal})
    position_change = (
        signal.diff().abs().fillna(0)
    )  # First row has no previous position

    my_ret = next_r * signal - position_change * cost

    pr = np.nansum(my_ret[my_ret > 0]) / np.abs(np.nansum(my_ret[my_ret < 0]))
    sharpe = np.nanmean(my_ret.dropna()) / np.nanstd(my_ret.dropna(), ddof=1)

    return my_ret, pr, sharpe, trade_signal
        


def compute_win_ratio(signal, log_return, horizon=4):
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


def generate_positions(signal, horizon=4):
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


def add_mas(df: pd.DataFrame, windows: List[int] | np.ndarray = np.arange(2, 25)):

    if "log_ret" not in df.columns:
        df["log_ret"] = np.log(df["close"]).diff()

    for window_length in windows:
        df[f"means_{window_length}"] = df["close"].rolling(window_length).mean()
        df[f"std_{window_length}"] = df["close"].rolling(window_length).std(ddof=1)

    return df


def add_rsis(df: pd.DataFrame, windows: List[int] | np.ndarray = np.arange(2, 25)):

    if "log_ret" not in df.columns:
        df["log_ret"] = np.log(df["close"]).diff()

    for window_length in windows:
        df[f"rsi_{window_length}"] = ta.rsi(df["close"], length=window_length)

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
