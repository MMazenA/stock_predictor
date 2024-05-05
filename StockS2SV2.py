import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import pandas_ta as ta

torch.manual_seed(12)


class StockLSTMPredictor:
    def __init__(
        self,
        ticker,
        start_date,
        end_date,
        sequence_length=10,
        hidden_size=35,
        learning_rate=0.001,
        epochs=75,
        interval="1d",
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.interval = interval
        self.data = None
        self.df = None
        self.df_scaled = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.criterion = None
        self.optimizer = None

    def download_data(self):
        data = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
        )
        self.data = data
        self.df = pd.DataFrame(data[["Close", "Volume", "Open", "High", "Low","Adj Close"]])
        self.df['RSI'] = ta.rsi(data.Close,length=15)
        self.df['EMAF'] = ta.ema(data.Close,length=20)
        self.df['EMAM'] = ta.ema(data.Close,length=100)
        self.df['EMAS'] = ta.ema(data.Close,length=150)
        self.df["p_change"] = self.df['Adj Close']-self.df.Open
        self.df["p_change"] = self.df['p_change'].shift(-1)
        self.df["up_down"] = [1 if self.df.p_change[i]>0 else 0 for i in range(len(self.df))]

    def preprocess_data(self):
        self.df.dropna(inplace=True)
        self.df.reset_index(inplace=True)
        self.df.drop(['Date'],axis=1,inplace=True)
        print(self.df.columns)
        self.df_scaled = self.scaler.fit_transform(self.df)
        train_size = int(len(self.df_scaled) * 0.80)

        train_data = self.df_scaled[:train_size]
        test_data = self.df_scaled[train_size - self.sequence_length :]
        
        self.X_train, self.y_train = self.create_sequences(train_data, self.sequence_length)
        self.X_test, self.y_test = self.create_sequences(test_data, self.sequence_length)

    def create_sequences(self, data, seq_length):
        sequences, targets = [], []
        for i in range(len(data) - seq_length):
            seq = torch.tensor(data[i : i + seq_length], dtype=torch.float32)
            target = torch.tensor(
                data[i + seq_length : i + seq_length + 1, 10:12], dtype=torch.float32
            ).squeeze() #last 2 things are targets
            sequences.append(seq)
            targets.append(target)
        return torch.stack(sequences), torch.stack(targets)

    def build_model(self):
        self.model = LSTMSeq2Seq(
            input_size=len(self.df.columns),
            hidden_size=self.hidden_size,
            output_size=len(self.df.columns),
        )
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

    def train_model(self, verbose=False):
        for epoch in range(self.epochs):
            self.model.train()
            for seq, labels in zip(self.X_train, self.y_train):
                self.optimizer.zero_grad()
                self.model.hidden_cell = (
                    torch.zeros(1, 1, self.model.hidden_size),
                    torch.zeros(1, 1, self.model.hidden_size),
                )
                # print(seq)
                y_pred = self.model(seq.unsqueeze(1))

                # last pchange
                predicted_close = y_pred[-1][10:12]

                # print("Prediction: ",predicted_close)
                # print("Real: ",labels)
                true_close = labels
                single_loss = self.criterion(predicted_close, true_close)
                single_loss.backward()
                self.optimizer.step()

            if verbose:
                print(f"Epoch [{epoch+1}], Loss: {single_loss.item():.6f}")
                train_rmse, test_rsme = self.evaluate_model()
                print(f"  Train: {train_rmse:.4f}")
                print(f"   Test: {test_rsme:.4f}")

    def evaluate_model(self):
        self.model.eval()
        self.train_predictions = []
        for seq in self.X_train:
            with torch.no_grad():
                self.model.hidden = (
                    torch.zeros(1, 1, self.model.hidden_size),
                    torch.zeros(1, 1, self.model.hidden_size),
                )
                y_pred = self.model(seq.unsqueeze(1))
                predicted_close = y_pred[-1, 10:12]  # last prediction for close price
                self.train_predictions.append(predicted_close.numpy())

        self.test_predictions = []
        for seq in self.X_test:
            with torch.no_grad():
                self.model.hidden = (
                    torch.zeros(1, 1, self.model.hidden_size),
                    torch.zeros(1, 1, self.model.hidden_size),
                )
                y_pred = self.model(seq.unsqueeze(1))
                predicted_close = y_pred[-1, 10:12]  # last prediction for close price
                self.test_predictions.append(predicted_close.numpy())

        self.train_predictions = np.array(self.train_predictions)
        self.y_train_inv = self.y_train.numpy().squeeze()
        self.test_predictions = np.array(self.test_predictions)
        self.y_test_inv = self.y_test.numpy().squeeze()

        train_rmse = np.sqrt(mean_squared_error(self.y_train_inv, self.train_predictions))
        test_rmse = np.sqrt(mean_squared_error(self.y_test_inv, self.test_predictions))

        return train_rmse, test_rmse


    def plot_predictions(self):
        train_predictions = self.train_predictions
        test_predictions = self.test_predictions

        plt.figure(figsize=(24, 12))

        # train predictions
        train_index = self.df.index[
            self.sequence_length : self.sequence_length + len(train_predictions)
        ]
        plt.plot(train_index, train_predictions, label="Train Predictions")

        test_index = self.df.index[
            self.sequence_length
            + len(train_predictions) : self.sequence_length
            + len(train_predictions)
            + len(test_predictions)
        ]
        plt.plot(test_index, test_predictions, label="Test Predictions")

        # actual prices
        plt.plot(self.df.index, self.df["Close"], label="Actual Prices", alpha=0.5)

        plt.title(f"{self.ticker} Stock Price Prediction (Seq2Seq)")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.legend()
        plt.show()

    def predict_future(self, initial_sequence, num_steps):
        future_predictions = []
        with torch.no_grad():
            current_sequence = initial_sequence
            for _ in range(num_steps):
                self.model.hidden_cell = (
                    torch.zeros(1, 1, self.model.hidden_size),
                    torch.zeros(1, 1, self.model.hidden_size),
                )
                next_pred = self.model(current_sequence.unsqueeze(1))
                predicted_close = next_pred[-1, 0]  # last prediction for close price

                future_predictions.append(predicted_close.item())


        future_predictions = self.scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        )

        return future_predictions


class LSTMSeq2Seq(nn.Module):
    def __init__(self, input_size=2, hidden_size=35, output_size=1):
        super(LSTMSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(self, input_seq):
        encoder_hidden = self.encoder(input_seq)
        output = self.decoder(input_seq[:, -1].unsqueeze(0), encoder_hidden)
        return output[-1]


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input_seq):
        _, hidden_state = self.lstm(input_seq)
        return hidden_state


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state):
        output, _ = self.lstm(input_seq, hidden_state)
        predictions = self.linear(output.squeeze(1))
        return predictions


def main():
    stock_predictor = StockLSTMPredictor(
        ticker="AAPL",
        start_date="2021-06-01",
        end_date="2025-10-31",
        learning_rate=0.0001,
        sequence_length=10,
        hidden_size=40,
        epochs=50,
        interval="1d",
    )

    stock_predictor.download_data()
    stock_predictor.preprocess_data()
    stock_predictor.build_model()
    stock_predictor.train_model(verbose=True)
    train_rmse, test_rmse = stock_predictor.evaluate_model()
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    # stock_predictor.plot_predictions()

    # Predict future
    initial_sequence = stock_predictor.X_test[-1]
    print(initial_sequence.size())
    num_future_steps = 1  # cannot predict further at the moment
    future_predictions = stock_predictor.predict_future(
        initial_sequence, num_future_steps
    )
    print("Future Predictions:")
    print(future_predictions)


if __name__ == "__main__":
    main()
