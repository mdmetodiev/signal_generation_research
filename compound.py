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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=64, output_dim=1, num_layers=1, dropout=0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,  # Dropout only if >1 layer
        )
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(
            x
        )  # lstm_out: (batch_size, seq_len, hidden_dim)
        # Take the output from the last timestep
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        last_out = self.batchnorm(last_out)
        last_out = self.dropout(last_out)
        output = self.fc(last_out)  # (batch_size, output_dim)
        # output = torch.sigmoid(output)  # Binary classification
        return output


# Define model
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 1),
            # nn.Sigmoid()  # Only use if you're not using BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x)


def continuous_signal(first_array: np.ndarray, N: int) -> np.ndarray:
    length = len(first_array)
    block_size = length // N
    remainder = length % N

    second_array = np.zeros_like(first_array)

    start = 0
    for i in range(N):
        end = start + block_size + (1 if i < remainder else 0)
        block = first_array[start:end]
        if np.any(block == 1):
            second_array[start:end] = 1
        start = end

    return second_array


def walk_forward_c(
    df: pd.DataFrame,
    train_period: pd.DateOffset = pd.DateOffset(months=6),
    test_period: pd.DateOffset = pd.DateOffset(months=6),
    N: int = 4,
    ntrain: int = 200,
    continuous: bool = False,
    cost: float = 1e-2,
    qmin: float = 0.05,
    qmax: float = 0.95,
    npca: int = 2,
    verbose: bool = False,
    plot: bool = False,
    save_path: str | None = None,
):

    df.index = pd.to_datetime(df["time"].values)

    results_list = []

    start_date = df.index.min()
    end_date = df.index.max()

    current_train_start = start_date
    current_train_end = current_train_start + train_period
    current_test_end = current_train_end + test_period

    pca = None
    BCmod = None

    all_signals = []
    raw_signals = []
    unprocessed_signals = []
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

        y_train = df_train["log_ret"].shift(-N) - df_train["log_ret"]
        y_train = y_train.dropna()

        X_train = X_train.iloc[:-N]
        X_train_clean = X_train.dropna()

        y_train_aligned = y_train.loc[X_train_clean.index]

        scaler = StandardScaler()
        X_train_zstd = scaler.fit_transform(X_train_clean)

        pca = PCA(n_components=npca)
        X_emb_train = pca.fit_transform(X_train_zstd)

        # model = RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=42)
        # model.fit(X_emb_train, y_train_aligned)

        ######## NN
        # _y_train_pred = model.predict(X_emb_train)

        scaler = StandardScaler()
        X_nn_train = scaler.fit_transform(X_emb_train)
        # X_nn_train = X_emb_train
        # print(len(X_emb_train), len(_y_train_pred))

        # X_nn_train_extra = _y_train_pred.copy()
        # X_nn_train_extra[_y_train_pred < 0] = 0

        # X_nn_train = np.column_stack([X_nn_train, X_nn_train_extra])

        # sig_nn_train, _, _ = learn.generate_signals(_y_train_pred, df_train, N=N, qmin=qmin, qmax=qmax)
        # sig_nn_train = pd.DataFrame({"tar":sig_nn_train})
        # pchange = sig_nn_train.diff().abs().fillna(0)

        # drag = pchange*cost
        # drag = drag[:-N]

        # nn_train_tar_args = np.where(y_train_aligned.values>drag.values)[0]
        # nn_train_tar = np.zeros_like(y_train_aligned)
        # nn_train_tar[nn_train_tar_args] = 1

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_nn_train)

        threshold = np.log(1 + cost)
        log_return = np.log(df_train["close"].shift(-N) / df_train["close"])
        log_return = log_return[:-N]
        const_q = np.quantile(log_return, 0.8)
        profitable = (log_return > const_q).astype(int)
        nn_train_tar = profitable.values

        sequence_length = 50  # Adjust based on your problem
        X_sequences = []
        y_sequences = []

        for i in range(len(X_nn_train) - sequence_length):
            X_sequences.append(X_scaled[i : i + sequence_length, :-1])  # Features
            y_sequences.append(nn_train_tar[i + sequence_length])  # Label (0 or 1)

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        X_tensor = torch.tensor(X_sequences, dtype=torch.float32).to(
            device
        )  # (num_samples, 50, num_features)
        y_tensor = (
            torch.tensor(y_sequences, dtype=torch.float32).unsqueeze(1).to(device)
        )  # (num_samples, 1)

        # # Convert to tensors
        # X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        # y_tensor = torch.tensor(nn_train_tar.values, dtype=torch.float32).unsqueeze(1).to(device)

        BCmod = LSTMModel(
            input_dim=X_tensor.shape[2], hidden_dim=64, num_layers=4, dropout=0.5
        )
        BCmod.to(device)

        if continuous == True:
            if c >= 1:
                print("loading weights")
                BCmod.load_state_dict(torch.load("model_weights.pth"))

        n_pos = (nn_train_tar == 1).sum()
        n_neg = (nn_train_tar == 0).sum()

        weight = torch.tensor([1 * n_neg / n_pos]).to(device)
        print(weight)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        optimizer = optim.Adam(BCmod.parameters(), lr=1e-3, weight_decay=1e-5)

        # Training loop
        for epoch in range(ntrain):
            BCmod.train()
            optimizer.zero_grad()
            outputs = BCmod(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        if verbose:
            print(f"Testing: {current_train_end} to {current_test_end}")

        if continuous == True:
            print("dumping weights")
            torch.save(BCmod.state_dict(), "model_weights.pth")

        X_test = utils.get_features_matrix(df_test)

        X_test = X_test.iloc[:-N]
        X_test_clean = X_test.dropna()
        test_idx = X_test_clean.index

        y_test = df_test["log_ret"].shift(-N) - df_test["log_ret"]
        y_test = y_test.loc[test_idx]  # Ensure alignment

        scaler = StandardScaler()
        X_test_zstd = scaler.fit_transform(X_test_clean)

        X_emb_test = pca.transform(X_test_zstd)
        # y_pred = model.predict(X_emb_test)

        # signal, up_q, low_q = learn.generate_signals(y_pred, df_test, N, qmin, qmax)

        # signal = utils.suppress_consecutive_repeats(signal)
        # signal = utils.generate_positions(signal)

        ######## NN

        scaler = StandardScaler()
        X_nn_test = scaler.fit_transform(X_emb_test)
        # X_nn_test = X_emb_test

        # X_nn_test_extra = y_pred.copy()
        # X_nn_test_extra[y_pred < 0] = 0
        # X_nn_test = np.column_stack([X_nn_test, X_nn_test_extra])

        log_return = np.log(df_test["close"].shift(-N) / df_test["close"])
        log_return_curr = np.log(df_test["close"].shift(-1) / df_test["close"])

        log_return = log_return[:-N]
        log_return_curr = log_return_curr[:-N]
        drag = (log_return - log_return_curr) * cost

        profitable = (log_return > np.quantile(log_return, 0.8)).astype(int)
        profitable = continuous_signal(profitable, N)
        nn_ground_truth = profitable

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        # optimizer,
        # mode='min',          # because we want to minimize the validation loss
        # factor=0.5,          # reduce LR by a factor of 0.5
        # patience=5,          # wait 5 epochs with no improvement
        # #verbose=True         # prints update messages
        # )

        # Standardize features
        scaler = StandardScaler()
        X_scaled_test = scaler.fit_transform(X_nn_test)

        X_sequences = []
        y_sequences = []

        for i in range(len(X_nn_test) - sequence_length):
            X_sequences.append(X_scaled_test[i : i + sequence_length, :-1])  # Features
            y_sequences.append(nn_ground_truth[i + sequence_length])  # Label (0 or 1)

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        X_tensor_test = torch.tensor(X_sequences, dtype=torch.float32).to(
            device
        )  # (num_samples, 50, num_features)
        y_tensor_test = (
            torch.tensor(y_sequences, dtype=torch.float32).unsqueeze(1).to(device)
        )

        # # Convert to tensors
        # X_tensor_test = torch.tensor(X_scaled_test, dtype=torch.float32).to(device).to(device)
        # y_tensor_test = torch.tensor(nn_ground_truth.values, dtype=torch.float32).unsqueeze(1).to(device).to(device)

        BCmod.eval()  # set BCmod to evaluation mode
        with torch.no_grad():
            logits = BCmod(X_tensor_test)  # raw probabilities (between 0 and 1)
            val_loss = criterion(logits, y_tensor_test)
            preds = torch.sigmoid(logits)

        preds = preds.cpu().squeeze(-1)

        class_chop = 0.6
        tar_args = np.where(preds.numpy() > class_chop)[0]

        tar = np.zeros(len(df_test))
        tar[:-N][tar_args] = 1

        # tar = utils.generate_positions(tar, horizon=N)
        # scheduler.step(val_loss)
        acc = accuracy_score(nn_ground_truth, tar[:-N])
        print(f"testing accuracy: {acc}, testing loss: {val_loss}")
        print(
            f"num real hits: {nn_ground_truth[nn_ground_truth==1].size}, num predicted: {np.where(preds>class_chop)[0].size}"
        )
        c_rep = classification_report(
            nn_ground_truth, tar[:-N], zero_division=0, output_dict=True
        )

        # print("up", up_q)
        # current_signals = pd.DataFrame({"signal": signal})
        # current_signals = current_signals.rolling(N).mean().fillna(0)
        # current_signals = current_signals.values * tar

        batch_tar = continuous_signal(tar, N)
        current_signals = pd.DataFrame({"signal": batch_tar})
        # current_signals = current_signals.rolling(N).mean().fillna(0)

        unprocessed_signals.append(tar)
        raw_signals.append(batch_tar)

        # _sig = utils.suppress_consecutive_repeats(current_signals["signal"].values)
        # current_signals["signal"] = utils.generate_positions(_sig, horizon=N)

        all_signals.append(current_signals.values)

        cl_precission = c_rep["1"]["precision"]
        print(f"class 1 accuracy: {cl_precission}")
        results_list.append(cl_precission)

        # plot first three tests

        # if plot:
        #     if c < 5:
        #         plt.plot(y_pred, y_test, "o", ms=2.3, alpha=0.32)
        #         plt.plot(y_test, y_test)
        #         plt.axhline(y=0, color="k", lw=1.23)
        #         for q in [up_q, low_q]:
        #             plt.axvline(x=q, color="r", lw=1.23)
        #         plt.title(f"testing: {current_train_end} - {current_test_end}")

        #     plt.show()

        # Roll forward one month
        current_train_start += test_period
        current_train_end += test_period
        current_test_end += test_period

        c = c + 1
        # if c >= 1:
        #     break

        # print("moving forward...")

    final_model = {}
    # final_model["uq"] = up_q
    # final_model["lq"] = low_q

    if pca and BCmod:
        final_model["pca"] = pca
        final_model["regressor"] = BCmod
        final_model["N"] = N

    if save_path:
        import pickle

        with open(f"{save_path}.pickle", "wb") as handle:
            pickle.dump(final_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # if isinstance(results_list[0], list):
    #     r = []
    #     for res in results_list:
    #         for val in res:
    #             r.append(val)
    # else:
    # mod_preds = np.array(results_list)

    mod_preds = results_list  # np.array(r)

    t = []
    t_raw = []
    for sig in all_signals:
        for val in sig:
            t.append(val)

    for sig in raw_signals:
        for val in sig:
            t_raw.append(val)

    t = np.array(t)
    t_raw = np.array(t)
    chopoff = start_date + train_period

    mask = df.index > chopoff

    df_trading = df[mask].copy()

    df_trading["signal"] = t
    df_trading["raw_signal"] = t_raw

    return final_model, mod_preds, raw_signals, unprocessed_signals, df_trading
