#! ~/anaconda3/bin/python
from binance.client import Client
import pandas as pd
import os


client = Client(api_key="", api_secret="")


symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1HOUR
start_str = "1 Jan 2020"
end_str = "23 May 2025"

klines = client.get_historical_klines(symbol, interval, start_str, end_str)


df = pd.DataFrame(
    klines,
    columns=[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_vol",
        "taker_buy_quote_vol",
        "ignore",
    ],
)


df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    df.to_csv(
        f"data/{symbol}_{interval}_{start_str.replace(" ", "_")}_{end_str.replace(" ", "_")}.csv",
        index=False,
    )
