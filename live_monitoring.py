#! ~/anaconda3/bin/python

from datetime import datetime, timedelta
import requests
import time


import utils
from binance.client import Client
import pickle
import utils
import pandas as pd
import time
import numpy as np


def get_hourly_prices(symbol, start_str):

    client = Client()  # No API key needed for public data
    klines = []
    interval = Client.KLINE_INTERVAL_1HOUR

    # Convert string start to timestamp (if needed)
    if isinstance(start_str, str):
        # start_ts = start_str
        start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    else:
        start_ts = start_str

    while True:
        new_klines = client.get_klines(
            symbol=symbol, interval=interval, startTime=start_ts, limit=1000
        )
        if not new_klines:
            break
        klines += new_klines

        last_open_time = new_klines[-1][0]
        start_ts = last_open_time + 1
        # time.sleep(0.5)

        # Optional stop if we're close to now
        if last_open_time > int(time.time() * 1000) - 3600 * 1000:
            break

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    client.close_connection()

    return df[["timestamp", "close", "volume"]]


def get_coinbase_price():
    now = datetime.now()

    # Ensure we never query the current or future incomplete candle
    # Subtract 1 second to push "now" into the previous hour safely
    safe_now = now - timedelta(seconds=1)

    # Align to the previous hour block
    end_time = safe_now.replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(hours=1)

    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    params = {
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "granularity": 3600,
    }

    response = requests.get(url, params=params)

    try:
        data = response.json()
    except Exception as e:
        raise ValueError(f"Could not parse JSON: {e}\nRaw response: {response.text}")

    if not isinstance(data, list) or not data:
        raise ValueError(
            f"No candle data returned from Coinbase API.\nStart: {start_time.isoformat()}, End: {end_time.isoformat()}"
        )

    candle = data[0]
    close_price = float(candle[4])
    volume = float(candle[5])
    timestamp = (
        pd.to_datetime(candle[0], unit="s").tz_localize("UTC").replace(tzinfo=None)
    )

    return close_price, timestamp, volume


def load_model():

    with open("rf_5days.pickle", "rb") as handle:
        new_model = pickle.load(handle)
    return new_model


def start_trading(current_hour, df):
    print("Now collecting live hourly close prices...")

    new_model = load_model()

    ts_list = []
    closing_prices = []

    last_price = None
    model_performance = {}

    model_performance["entry_uq_signals"] = []
    model_performance["entry_lq_signals"] = []
    model_performance["entry_uq_ts"] = []
    model_performance["exit_uq_ts"] = []
    model_performance["entry_lq_ts"] = []
    model_performance["exit_lq_ts"] = []
    model_performance["uq_price"] = []
    model_performance["lq_price"] = []

    model_performance["model_hourly_values"] = []
    model_performance["model_hourly_times"] = []
    model_performance["measured_timestamps"] = []
    model_performance["measured_closing_prices"] = []

    N = new_model["N"]
    model_performance["uq"] = new_model["uq"]
    model_performance["lq"] = new_model["lq"]
    model_performance["N"] = N
    upper_q = new_model["uq"]
    lower_q = new_model["lq"]

    current_hour = datetime.now().hour
    try:
        while True:

            now = datetime.now()

            if now.hour != current_hour:
                print("\n =======================")
                print("adding new price")
                print("\n =======================")
                last_price, ts, volume = get_coinbase_price()
                closing_prices.append(last_price)
                ts_list.append(ts)

                print(
                    f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Live hourly close: {last_price}"
                )
                current_hour = now.hour
                _temp = pd.DataFrame(
                    [{"timestamp": ts, "close": last_price, "volume": volume}]
                )
                df_new = pd.concat([df, _temp])
                print("adding geometry...")

                df_new = utils.add_rsis(df)

                model_performance["dataframe"] = df_new
                model_performance["measured_timestamps"].append(ts)
                model_performance["model_hourly_times"].append(current_hour)
                model_performance["measured_closing_prices"].append(closing_prices)

                print("captured timestamp: ", df_new["timestamp"].iloc[-1])

                X_features = utils.get_features_matrix(df_new)

                X_last = X_features.iloc[-1].values
                X_last_emb = new_model["pca"].transform(X_last.reshape(1, -1))
                mod_value = new_model["regressor"].predict(X_last_emb)[0]
                model_performance["model_hourly_values"].append(mod_value)

                print("\n============PREDICTIONS===============")
                print(
                    f"model precition: {mod_value:.4f}, upper q: {upper_q:.4f}, lower_q: {lower_q:.4f}"
                )
                print(f"last price: {last_price}")

                if mod_value > upper_q:
                    print(
                        f"model above upper quantile - price increase in next {N} hours"
                    )

                    entry_time = now

                    model_performance["entry_uq_signals"].append(1)
                    model_performance["entry_uq_ts"].append(entry_time)
                    model_performance["exit_uq_ts"].append(
                        entry_time + timedelta(hours=N)
                    )
                    model_performance["uq_price"].append(last_price)

                if mod_value < lower_q:
                    print(
                        f"model below lower quantile - price decrease in next {N} hours"
                    )

                    entry_time = now

                    model_performance["entry_lq_signals"].append(1)
                    model_performance["entry_lq_ts"].append(entry_time)
                    model_performance["exit_lq_ts"].append(
                        entry_time + timedelta(hours=N)
                    )
                    model_performance["lq_price"].append(last_price)

    except KeyboardInterrupt:

        print("\nStopped by user.")
        print(f"Total closing prices collected: {len(closing_prices)}")
        _now = str(now).replace(" ", "_").replace(":", "_").replace(".", "_")
        _filename = f"model_performance_{_now}"

        with open(_filename, "wb") as handle:
            pickle.dump(model_performance, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"model performance dumped here : {_filename}")


if __name__ == "__main__":

    df = get_hourly_prices("BTCUSDT", "1 Jan 2025")
    print("obtained newest data")

    df = utils.add_rsis(df)
    current_hour = datetime.now().hour

    start_trading(current_hour, df)
