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
start_date = "2023-01-01"
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
print(df.dropna(inplace=True, axis=0))
df.reset_index(inplace=True)
df.drop(["Close", "Date"], axis=1, inplace=True)
print(df.shape)
print(df.tail())


df_unscaled = df.copy()
scaler = MinMaxScaler(feature_range=(-1, 1))
df = np.array(df)
df = scaler.fit_transform(df)


def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        seq = torch.tensor(data[i : i + seq_length], dtype=torch.float32)
        target = torch.tensor(data[i + 1 : i + seq_length + 1], dtype=torch.float32)
        # seq = torch.transpose(seq,0,1)
        # target = torch.transpose(target,0,1)
        sequences.append(seq)
        targets.append(target)
    return torch.stack(sequences).to(device), torch.stack(targets).to(device)


seq_length = 1
train_size = int(len(df) * 0.80)
train_data = df[0:train_size]
test_data = df[train_size : len(df)]

print(train_data.shape)
print(test_data.shape)

X_train, Y_train = create_sequences(train_data, seq_length)
X_test, Y_test = create_sequences(test_data, seq_length)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, embedding_dim
        self.num_layers = 3
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.35,
        )

    def forward(self, x):

        x = x.unsqueeze(2)
        h_1 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        )

        c_1 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        )

        x = torch.transpose(x, 1, 2)

        x, (hidden, cell) = self.rnn1(x, (h_1, c_1))

        return x, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        hidden = hidden[2:3, :, :]

        # print("hidden size is",hidden.size())

        # repeat decoder hidden state src_len times
        # hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        hidden = hidden.repeat(1, src_len, 1)

        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # print("encode_outputs size after permute is:",encoder_outputs.size())

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class AttentionDecoder(nn.Module):
    def __init__(
        self, seq_len, attention, input_dim=64, n_features=1, encoder_hidden_state=512
    ):
        super(AttentionDecoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = input_dim, n_features
        self.attention = attention

        self.rnn1 = nn.LSTM(
            # input_size=1,
            input_size=encoder_hidden_state
            + n_features,  # Encoder Hidden State + One Previous input
            hidden_size=input_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.35,
        )

        self.output_layer = nn.Linear(self.hidden_dim * 2, n_features)

    def forward(self, x, input_hidden, input_cell, encoder_outputs):

        a = self.attention(input_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)
        # x = x.reshape((1, 1, 1))
        x = x.unsqueeze(1)
        rnn_input = torch.cat((x, weighted), dim=2)

        # x, (hidden_n, cell_n) = self.rnn1(x,(input_hidden,input_cell))
        x, (hidden_n, cell_n) = self.rnn1(rnn_input, (input_hidden, input_cell))

        output = x.squeeze(0)
        weighted = weighted.squeeze(0)

        x = self.output_layer(torch.cat((output, weighted), dim=1))
        return x, hidden_n, cell_n


class Seq2Seq(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64, output_length=28):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim=embedding_dim).to(
            device
        )
        self.attention = Attention(512, 512)
        self.output_length = output_length
        self.decoder = AttentionDecoder(
            seq_len, self.attention, embedding_dim, n_features
        ).to(device)

    def forward(self, x, prev_y):

        encoder_output, hidden, cell = self.encoder(x)

        # Prepare place holder for decoder output
        targets_ta = []
        # prev_output become the next input to the LSTM cell
        prev_output = prev_y

        # itearate over LSTM - according to the required output days
        for out_days in range(self.output_length):

            prev_x, prev_hidden, prev_cell = self.decoder(
                prev_output, hidden, cell, encoder_output
            )
            hidden, cell = prev_hidden, prev_cell
            prev_output = prev_x

            targets_ta.append(prev_x.reshape(1, -1))  # Reshape to [1, n_features]

        targets = torch.stack(targets_ta)

        return targets


n_features = 21
model = Seq2Seq(seq_length, n_features, embedding_dim=512)
model = model.to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=4e-3, weight_decay=1e-5)
criterion = torch.nn.MSELoss().to(device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=5e-3, eta_min=1e-8, last_epoch=-1
)


def train_model(model, TrainX, Trainy, ValidX, Validy, seq_length, n_epochs):

    history = dict(train=[], val=[])

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    mb = master_bar(range(1, n_epochs + 1))

    for epoch in mb:
        model = model.train()

        train_losses = []
        for i in progress_bar(range(TrainX.size()[0]), parent=mb):
            seq_inp = TrainX[i].to(device)
            seq_true = Trainy[i].to(device)

            optimizer.zero_grad()

            seq_pred = model(seq_inp, seq_true)

            loss = criterion(seq_pred[:, :, -2:], seq_true[:, -2:])

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for i in progress_bar(range(ValidX.size()[0]), parent=mb):
                seq_inp = ValidX[i, :, :].to(device)
                seq_true = Validy[i, :, :].to(device)

                seq_pred = model(seq_inp, seq_true)
                loss = criterion(seq_pred[:, :, -2:], seq_true[:, -2:])
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("saved best model epoch:", epoch, "val loss is:", val_loss)

        print(f"Epoch {epoch}: train loss {train_loss} val loss {val_loss}")
        scheduler.step()
    # model.load_state_dict(best_model_wts)
    return model.eval(), history


model, history = train_model(
    model,
    X_train,
    Y_train,
    X_test,
    Y_test,
    seq_length,
    n_epochs=30,  ## Training only on 30 epochs to save GPU time
)
