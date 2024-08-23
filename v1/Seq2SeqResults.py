import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import pandas_ta as ta
import random
import os
from torch.autograd import Variable
import torch.nn.functional as F
from fastprogress import master_bar, progress_bar

# from Seq2Seq import Encoder, AttentionDecoder, Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SEED = 11


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(SEED)


ticker = "AAPL"
start_date = "2024-01-01"
end_date = "2025-10-31"
interval = "1d"


# data = yf.download("AAPL")
data = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    #  interval=interval,
)
df = pd.DataFrame(data[["Close", "Volume", "Open", "High", "Low", "Adj Close"]])
df["RSI"] = ta.rsi(data.Close, length=15)
df["EMAF"] = ta.ema(data.Close, length=20)
df["EMAM"] = ta.ema(data.Close, length=100)
df["EMAS"] = ta.ema(data.Close, length=150)
a = ta.volume.eom(data.High, data.Low, data.Close, data.Volume)
df = df.join(a)

a = ta.stoch(data.High, data.Low, data.Close, length=20)
df = df.join(a)

a = ta.macd(data.Close)
df = df.join(a)

a = ta.adx(df["High"], df["Low"], df["Close"], length=14)
df = df.join(a)

a = ta.volume.adosc(df["High"], df["Low"], df["Close"], data.Volume)
df = df.join(a)

df["p_change"] = df["Adj Close"] - df.Open
df["p_change"] = df["p_change"].shift(-1)
df["up_down"] = [1 if df.p_change[i] > 0 else 0 for i in range(len(df))]
df.dropna(inplace=True)
df.reset_index(inplace=True)
print(df.tail(100))
df.drop(["Close", "Date"], axis=1, inplace=True)
last_row = df.tail(2)
print(last_row)
df.drop(df.tail(1).index, inplace=True)

scaler = MinMaxScaler(feature_range=(-1, 1))
df = np.array(df)
df = scaler.fit_transform(df)

seq_length = 10
n_features = 21


model = torch.load("best_model.pt")
model.eval()

with torch.no_grad():
    seq_inp = last_row.to(device)
    seq_pred = model(seq_inp, seq_inp)


print(data_predict=scaler.inverse_transform(seq_pred.cpu().numpy()))
