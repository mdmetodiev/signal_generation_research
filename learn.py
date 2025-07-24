#! ~/anaconda3/envs/alpha/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from typing import Tuple, Any, List
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score

from joblib import Parallel, delayed
from itertools import product


from typing import Protocol


class RegressorProtocol(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressorProtocol": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


def train_model_in_sample(
    df: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    N: int = 4,
    npca: int = 4,
) -> Tuple[PCA, RegressorProtocol, np.ndarray, np.ndarray]:

    features = utils.get_features_matrix(df)

    scaler = StandardScaler()
    zsc_f = scaler.fit_transform(features)

    zsc_f = pd.DataFrame(zsc_f, columns=features.columns)

    X = zsc_f.copy()
    X_t = X[:-N]
    y = y[:-N]

    pca = PCA(n_components=npca)
    X_emb = pca.fit_transform(X_t)

    ridge = RidgeCV(alphas=np.logspace(-4, 4, 20), cv=TimeSeriesSplit(n_splits=5))
    model = ridge.fit(X_emb, y)

    y_pred = model.predict(X_emb)

    return pca, model, y_pred, np.array(y)


def train_model_on_subwindows(
    df: pd.DataFrame,
    N: int,
    window_size: int,
    qvals: Tuple[float, float] = (0.05, 0.95),
    npca: int = 4,
    vali_split: float = 0.05,
) -> Tuple[
    PCA | None,
    RegressorProtocol | None,
    StandardScaler | None,
    List[np.ndarray],
    List[np.ndarray],
    pd.Series,
    float | None,
    float | None,
    List[np.ndarray],
    List[Tuple[float, float]],
    List[pd.Series],
]:
    """
    Trains models on fixed-length subwindows of the in-sample data, tests within the same window.
    Returns predictions, ground truth, and sharpe ratios or error metrics.
    """

    forecast_horizon = int(vali_split * window_size)
    step = forecast_horizon

    preds_all = []
    targets_all = []

    pca = None
    model = None
    scaler = None

    up_q = None
    low_q = None

    subsignals = []
    subqs = []
    logrs = []
    rs = df["log_ret"]

    signal = np.zeros(len(df))
    qmin, qmax = qvals

    for i in range(window_size, len(df) - forecast_horizon + 1, step):

        subdf_train = df.iloc[i - window_size : i]
        y_train = np.log(subdf_train["close"].shift(-N) / subdf_train["close"])[:-N]

        features = utils.get_features_matrix(subdf_train).iloc[:-N]

        scaler = StandardScaler()
        zsc_f = scaler.fit_transform(features)
        X_train = pd.DataFrame(zsc_f, columns=features.columns)

        subdf_test = df.iloc[i : i + forecast_horizon]
        y_test = np.log(subdf_test["close"].shift(-N) / subdf_test["close"])[:-N]

        features_test = utils.get_features_matrix(subdf_test).iloc[:-N]

        zsc_f_test = scaler.transform(features_test)
        X_test = pd.DataFrame(zsc_f_test, columns=features.columns)

        pca = PCA(n_components=npca)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        ridge = RidgeCV(alphas=np.logspace(-4, 4, 20), cv=TimeSeriesSplit(n_splits=5))
        model = ridge.fit(X_train_pca, y_train)
        y_pred_train = model.predict(X_train_pca)
        y_pred_test = model.predict(X_test_pca)

        preds_all.append(y_pred_test)
        targets_all.append(y_test)

        up_q = float(np.quantile(y_pred_train, qmax))
        low_q = float(np.quantile(y_pred_train, qmin))

        long_args = np.where(y_pred_test > up_q)[0]
        short_args = np.where(y_pred_test < low_q)[0]

        test_len = len(y_pred_test)
        signal_indices = np.arange(i, i + test_len)

        signal[signal_indices[long_args]] = 1
        signal[signal_indices[short_args]] = -1

        subsignals.append(signal[signal_indices])
        subqs.append((up_q, low_q))
        logrs.append(rs.iloc[signal_indices])

    last_chunk = df.iloc[-window_size:]
    features_last = utils.get_features_matrix(last_chunk).iloc[:-N]
    y_last_train = np.log(last_chunk["close"].shift(-N) / last_chunk["close"])[:-N]

    scaler_last = StandardScaler()
    zsc_f_last = scaler_last.fit_transform(features_last)
    zsc_f_last = pd.DataFrame(zsc_f_last, columns=features_last.columns)

    pca_last = PCA(n_components=npca)
    X_last_pca = pca_last.fit_transform(zsc_f_last)

    ridge = RidgeCV(alphas=np.logspace(-4, 4, 20), cv=TimeSeriesSplit(n_splits=5))
    model_last = ridge.fit(X_last_pca, y_last_train)
    y_pred_last = model_last.predict(X_last_pca)

    up_q_last = float(np.quantile(y_pred_last, qmax))
    low_q_last = float(np.quantile(y_pred_last, qmin))

    signal_df = pd.DataFrame({"signal": signal})
    signal_final = signal_df["signal"].rolling(N, min_periods=1).mean().fillna(0)
    signal_final.index = df.index

    return (
        pca_last,
        model_last,
        scaler_last,
        preds_all,
        targets_all,
        signal_final,
        up_q_last,
        low_q_last,
        subsignals,
        subqs,
        logrs,
    )


def optimise_strategy(
    df: pd.DataFrame,
    tw_vals: List[int] | np.ndarray,
    look_ahead_vals: np.ndarray,
    npca_vals: np.ndarray,
    vali_split: float = 0.05,
    cost: float = 0.00,
) -> Tuple[dict, int]:

    optimisation_dict = {}
    optimisation_dict["sharpe_validate"] = []
    optimisation_dict["profit_ratio_validate"] = []
    optimisation_dict["strategy_returns"] = []

    optimisation_dict["model_outputs"] = []
    optimisation_dict["model_signals"] = []
    optimisation_dict["model_params_list"] = []

    optimisation_dict["vali_performance_subsigs"] = []
    optimisation_dict["vali_performance_logrs"] = []
    optimisation_dict["vali_performance_subqs"] = []

    optimisation_dict["pr_distros"] = []
    optimisation_dict["sharpe_distros"] = []

    for time_val in tw_vals:
        for i in range(look_ahead_vals.size):
            for j in range(npca_vals.size):

                N = look_ahead_vals[i]
                npca = npca_vals[j]

                (
                    _pca,
                    _model,
                    _scaler,
                    _y_pred,
                    _y,
                    signal,
                    up_q,
                    low_q,
                    subsigs,
                    subqs,
                    logrs,
                ) = train_model_on_subwindows(
                    df,
                    N=N,
                    npca=npca,
                    window_size=time_val,
                    vali_split=vali_split,
                )

                sharpe_distro = []
                pr_distro = []

                for sig, logr in zip(subsigs, logrs):

                    sig = pd.Series(sig)
                    _, _pr, _sharpe = utils.compute_returns(sig, logr, cost=cost)
                    sharpe_distro.append(_sharpe * np.sqrt(365 * 24))
                    pr_distro.append(_pr)

                optimisation_dict["pr_distros"].append(pr_distro)
                optimisation_dict["sharpe_distros"].append(sharpe_distro)

                _strat_ret, _pr, _sharpe = utils.compute_returns(
                    signal, df["log_ret"], cost=cost
                )

                optimisation_dict["sharpe_validate"].append(_sharpe * np.sqrt(365 * 24))
                optimisation_dict["profit_ratio_validate"].append(_pr)
                optimisation_dict["model_outputs"].append(
                    (_pca, _model, _scaler, _y_pred, _y)
                )
                optimisation_dict["model_params_list"].append((N, npca, time_val))
                optimisation_dict["model_signals"].append((signal, up_q, low_q))
                optimisation_dict["vali_performance_subsigs"].append(subsigs)
                optimisation_dict["vali_performance_logrs"].append(logrs)
                optimisation_dict["vali_performance_subqs"].append(subqs)

                optimisation_dict["strategy_returns"].append(_strat_ret)

    average_profit_ratios = np.array(
        [
            np.nanmean(pr) if not np.all(np.isnan(pr)) else np.nan
            for pr in optimisation_dict["pr_distros"]
        ]
    )

    if np.all(np.isnan(average_profit_ratios)):
        best_idx = -1
        optimisation_dict["best_pr"] = np.nan
    else:
        best_idx = int(np.nanargmax(average_profit_ratios))
        optimisation_dict["best_pr"] = average_profit_ratios[best_idx]

    return optimisation_dict, best_idx


def _output_signal(
    opti_dict: dict, df_test: pd.DataFrame, idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    pca, model, scaler, _, _ = opti_dict["model_outputs"][idx]
    _, up_q, low_q = opti_dict["model_signals"][idx]
    N_best, _, _ = opti_dict["model_params_list"][idx]

    y_target_test = np.log(df_test["close"].shift(-N_best) / df_test["close"])[:-N_best]

    y_target_test = -1 * y_target_test

    feats_test = utils.get_features_matrix(df_test).iloc[:-N_best]
    Z_test = pd.DataFrame(scaler.transform(feats_test), columns=feats_test.columns)
    X_test_pca = pca.transform(Z_test)
    y_pred_test = model.predict(X_test_pca)

    signal_slice = np.zeros(len(df_test))
    long_idx = np.where(y_pred_test > up_q)[0]
    short_idx = np.where(y_pred_test < low_q)[0]

    signal_slice[long_idx] = 1
    signal_slice[short_idx] = -1

    sig_df = (
        pd.Series(signal_slice)
        .rolling(window=N_best, min_periods=1)
        .mean()
        .fillna(0)
        .to_numpy()
    )

    return sig_df, y_pred_test, y_target_test


def _process_window(
    df: pd.DataFrame,
    i: int,
    opt_period: int,
    test_period: int,
    look_ahead_vals: np.ndarray,
    npca_vals: np.ndarray,
    lookback_vals: list[int],
    vali_split: float,
    cost: float,
) -> Tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Runs one optimisation + test on window starting at index i.
    Returns (i, y_pred_test, y_true_test, signal_slice).
    """

    df_train = df.iloc[i - opt_period : i]
    df_test = df.iloc[i : i + test_period]

    opti_dict, pos_idx = optimise_strategy(
        df_train,
        tw_vals=lookback_vals,
        look_ahead_vals=look_ahead_vals,
        npca_vals=npca_vals,
        vali_split=vali_split,
        cost=cost,
    )

    sig_arr, y_pred_test, y_target_test = _output_signal(opti_dict, df_test, pos_idx)

    if pos_idx == -1:
        sig_arr = np.zeros(len(sig_arr))

    return i, y_pred_test, y_target_test, sig_arr


def walk_forward_parallel(
    df: pd.DataFrame,
    look_ahead_vals: np.ndarray,
    npca_vals: np.ndarray,
    lookback_vals: list[int],
    optimisation_period: int = 90,
    test_period: int = 3,
    vali_split: float = 0.05,
    cost: float = 0.00,
    n_jobs: int = -1,
) -> Tuple[pd.Series, np.ndarray | List[List[float]], np.ndarray | List[List[float]]]:

    opt_h = 24 * optimisation_period
    tst_h = 24 * test_period

    starts = list(range(opt_h, len(df) - tst_h + 1, tst_h))

    # parallel dispatch
    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_process_window)(
            df,
            i,
            opt_h,
            tst_h,
            look_ahead_vals,
            npca_vals,
            lookback_vals,
            vali_split,
            cost,
        )
        for i in starts
    )

    outputs.sort(key=lambda x: x[0])

    # pre‑allocate full arrays
    full_signal = np.zeros(len(df))

    all_preds = []
    all_truth = []

    for i, y_pred, y_true, sig_slice in outputs:

        full_signal[i : i + len(sig_slice)] = sig_slice

        all_preds.append(y_pred)
        all_truth.append(y_true)

    # flat_preds = np.concatenate(all_preds)
    # flat_truth = np.concatenate(all_truth)

    full_signal = pd.Series(full_signal)

    return (
        full_signal,
        all_preds,
        all_truth,
    )


def walk_forward(
    df: pd.DataFrame,
    look_ahead_vals: np.ndarray,
    npca_vals: np.ndarray,
    lookback_vals: List[int] = [5],
    optmisiation_period: int = 90,
    test_period: int = 3,
    vali_split: float = 0.1,
    cost: float = 0.00,
    verbose: bool = False,
    plot: bool = False,
    save_path: str | None = None,
):
    signal = np.zeros(len(df))

    df.index = pd.to_datetime(df["time"].values)

    results_list = []
    y_test_gt = []

    up_q = None
    low_q = None
    pca = None
    model = None
    N_best = None

    optmisiation_period = 24 * optmisiation_period
    test_period = 24 * test_period

    all_signals = []

    for i in range(optmisiation_period, len(df) - test_period + 1, test_period):

        current_signal = np.zeros(len(df))

        dates_train = df.index[i - optmisiation_period : i]
        current_train_start = dates_train[0]
        current_train_end = dates_train[-1]

        dates_test = df.index[i : i + test_period]
        current_test_start = dates_test[0]
        current_test_end = dates_test[-1]

        if verbose:
            print(f"Training: {current_train_start} to {current_train_end}...")

        df_train = df.iloc[i - optmisiation_period : i]
        df_test = df.iloc[i : i + test_period]

        opti_dict, best_idx = optimise_strategy(
            df_train,
            tw_vals=lookback_vals,
            look_ahead_vals=look_ahead_vals,
            npca_vals=npca_vals,
            vali_split=vali_split,
            cost=cost,
        )

        pca, model, scaler, y_pred_train, y_train = opti_dict["model_outputs"][best_idx]

        _, up_q, low_q = opti_dict["model_signals"][best_idx]

        best_params = opti_dict["model_params_list"][best_idx]
        N_best, npca_best, lookback = best_params

        if verbose:
            print(f"Testing: {current_train_end} to {current_test_end}")

        y_target_test = np.log(df_test["close"].shift(-N_best) / df_test["close"])

        features_test = utils.get_features_matrix(df_test)
        features_test_aligned = features_test.iloc[:-N_best]

        zsc_f_test = scaler.transform(features_test_aligned)
        zsc_f_test = pd.DataFrame(zsc_f_test, columns=features_test.columns)

        X_test = pca.transform(zsc_f_test)
        y_target_test = y_target_test[:-N_best]

        y_pred_test = model.predict(X_test)

        long_args = np.where(y_pred_test > up_q)[0]
        short_args = np.where(y_pred_test < low_q)[0]

        test_len = len(y_pred_test)
        signal_indices = np.arange(i, i + test_len)

        current_signal[signal_indices[long_args]] = 1
        current_signal[signal_indices[short_args]] = -1

        current_test_signal = pd.DataFrame({"signal": current_signal})
        current_signal_smooth = (
            current_test_signal["signal"]
            .rolling(N_best, min_periods=1)
            .mean()
            .fillna(0)
        )

        signal[signal_indices] = current_signal_smooth.values

        all_signals.append(current_signal_smooth.values)
        results_list.append(y_pred_test)
        y_test_gt.append(y_target_test)

        # plot first five tests

        if plot:
            r2val = r2_score(y_true=y_target_test, y_pred=y_pred_test)
            if r2val > 0.3:

                print(
                    f"High performance period: {current_train_end} - {current_test_end}"
                )

                fig, axs = plt.subplots(1, 2, figsize=(14, 4))
                axs[0].plot(y_target_test, label="ground truth")
                axs[0].plot(y_pred_test, label="predicted")

                axs[0].axhline(y=0, color="k", lw=1.23)

                for q in [up_q, low_q]:
                    axs[0].axhline(y=q, color="r", lw=1.23)
                axs[0].set_title(f"testing: {current_train_end} - {current_test_end}")
                axs[0].legend()

                axs[1].plot(y_pred_test, y_target_test, "o", ms=2.3, alpha=0.32)
                axs[1].set_xlabel("predicted")
                axs[1].set_ylabel("ground truth")
                axs[1].axhline(y=0, color="k", lw=1.23)

                for q in [up_q, low_q]:
                    axs[1].axvline(x=q, color="r", lw=1.23)
                axs[1].set_title(f"testing: {current_train_end} - {current_test_end}")

            plt.show()
            plt.tight_layout()
            plt.close()

    signal = pd.Series(signal)
    signal.index = df.index

    final_model = {}
    final_model["uq"] = up_q
    final_model["lq"] = low_q

    final_model["pca"] = pca
    final_model["regressor"] = model
    final_model["N"] = N_best

    if save_path:
        import pickle

        with open(f"{save_path}.pickle", "wb") as handle:
            pickle.dump(final_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # r = []
    # for res in results_list:
    #     for val in res:
    #         r.append(val)

    mod_preds = results_list  # np.array(r)

    # t = []
    # for sig in all_signals:
    #     for val in sig:
    #         t.append(val)

    # t = np.array(t)
    # chopoff = start_date + train_period

    # mask = df.index > chopoff

    # df_trading = df[mask].copy()

    # df_trading["signal"] = t

    return final_model, mod_preds, y_test_gt, signal
