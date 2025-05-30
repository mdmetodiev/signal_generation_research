#! ~/anaconda3/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from typing import Tuple, Any

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor


def train_model_on_all_data(
    df_f: pd.DataFrame, y, N=4
) -> Tuple[PCA, LinearRegression, np.ndarray, np.ndarray]:

    X = df_f
    X_t = X[:-N]
    y = y[:-N]

    pca = PCA()
    X_emb = pca.fit_transform(X_t)

    model = LinearRegression().fit(X_emb, y)

    y_pred = model.predict(X_t)

    return pca, model, y_pred, y


def train_rf(
    df: pd.DataFrame, y, N=4
) -> Tuple[PCA, RandomForestRegressor, np.ndarray, np.ndarray]:

    X = utils.get_features_matrix(df)
    X_t = X[:-N]

    y = y[:-N]

    pca = PCA(n_components=4)
    X_emb = pca.fit_transform(X_t)

    # scaler = StandardScaler()
    # X_emb = scaler.fit_transform(X_emb)

    # X_emb = np.cumsum(X_emb, axis=0)

    # print("using cumulative sum")
    # Initialize and train model
    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=1, n_jobs=-1, random_state=42
    )
    model.fit(X_emb, y)

    # Predict
    y_pred = model.predict(X_emb)

    return pca, model, y_pred, y


def generate_signals(
    y_pred: np.ndarray, df: pd.DataFrame, N: int, qmin: float = 0.05, qmax: float = 0.95
) -> Tuple[np.ndarray, float, float]:

    signal = np.zeros(len(df))

    assert len(signal[:-N]) == len(y_pred)

    # Apply quantile cuts
    up_q = float(np.quantile(y_pred, qmax))
    low_q = float(np.quantile(y_pred, qmin))

    long_args = np.where(y_pred > up_q)[0]
    short_args = np.where(y_pred < low_q)[0]

    signal[:-N][long_args] = 1
    signal[:-N][short_args] = -1

    # signal = utils.suppress_consecutive_repeats(signal)

    return signal, up_q, low_q


### walk forward


def walk_forward(
    df: pd.DataFrame,
    train_period: pd.DateOffset = pd.DateOffset(months=6),
    test_period: pd.DateOffset = pd.DateOffset(months=6),
    N: int = 4,
    qmin: float = 0.05,
    qmax: float = 0.95,
    npca: int = 2,
    verbose: bool = False,
    plot: bool = False,
    save_path: str | None = None,
):

    df.index = pd.to_datetime(df["time"].values)

    results_list = []
    y_test_gt = []

    up_q = None
    low_q = None
    pca = None
    model = None

    start_date = df.index.min()
    end_date = df.index.max()

    current_train_start = start_date
    current_train_end = current_train_start + train_period
    current_test_end = current_train_end + test_period

    all_signals = []
    c = 0
    while current_test_end < end_date + test_period:

        if verbose:
            print(f"Training: {current_train_start} to {current_train_end}...")

        train_mask = (df.index >= current_train_start) & (df.index < current_train_end)
        df_train = df.loc[train_mask]

        test_mask = (df.index >= current_train_end) & (
            df.index < min(current_test_end, end_date)
        )
        df_test = df.loc[test_mask]

        X_train = utils.get_features_matrix(df_train)

        y_train = np.log(df_train["close"].shift(-N) / df_train["close"])
        y_train = y_train.dropna()

        X_train = X_train.iloc[:-N]
        X_train_clean = X_train.dropna()

        y_train_aligned = y_train.loc[X_train_clean.index]

        scaler = StandardScaler()
        X_train_zstd = scaler.fit_transform(X_train_clean)

        pca = PCA(n_components=npca)
        X_emb_train = pca.fit_transform(X_train_zstd)

        model = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=1, n_jobs=-1, random_state=42
        )
        model.fit(X_emb_train, y_train_aligned)

        if verbose:
            print(f"Testing: {current_train_end} to {current_test_end}")

        X_test = utils.get_features_matrix(df_test)

        X_test = X_test.iloc[:-N]
        X_test_clean = X_test.dropna()
        test_idx = X_test_clean.index

        y_test = np.log(df_test["close"].shift(-N) / df_test["close"])
        y_test = y_test.loc[test_idx]  # Ensure alignment

        scaler = StandardScaler()
        X_test_zstd = scaler.fit_transform(X_test_clean)

        X_emb_test = pca.transform(X_test_zstd)

        y_pred = model.predict(X_emb_test)

        signal, up_q, low_q = generate_signals(y_pred, df_test, N, qmin, qmax)

        # signal = utils.suppress_consecutive_repeats(signal)
        # signal = utils.generate_positions(signal)

        # print("up", up_q)
        # signal =
        current_signals = pd.DataFrame({"signal": signal})
        current_signals = current_signals.rolling(N).mean().fillna(0)

        all_signals.append(current_signals.values)
        results_list.append(y_pred)
        y_test_gt.append(y_test)

        # plot first three tests

        if plot:
            if c < 5:
                plt.plot((y_pred), "o", ms=2.3, alpha=0.32)
                plt.plot((y_test.values))
                plt.axhline(y=0, color="k", lw=1.23)

                for q in [up_q, low_q]:
                    plt.axhline(y=q, color="r", lw=1.23)
                plt.title(f"testing: {current_train_end} - {current_test_end}")
            plt.show()
        c = c + 1

        # Roll forward one month
        current_train_start += test_period
        current_train_end += test_period
        current_test_end += test_period

        # print("moving forward...")

    final_model = {}
    final_model["uq"] = up_q
    final_model["lq"] = low_q

    final_model["pca"] = pca
    final_model["regressor"] = model
    final_model["N"] = N

    if save_path:
        import pickle

        with open(f"{save_path}.pickle", "wb") as handle:
            pickle.dump(final_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # r = []
    # for res in results_list:
    #     for val in res:
    #         r.append(val)

    mod_preds = results_list  # np.array(r)

    t = []
    for sig in all_signals:
        for val in sig:
            t.append(val)

    t = np.array(t)
    chopoff = start_date + train_period

    mask = df.index > chopoff

    df_trading = df[mask].copy()

    df_trading["signal"] = t

    return final_model, mod_preds, y_test_gt, df_trading
