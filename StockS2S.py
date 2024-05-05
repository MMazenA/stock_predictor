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
        self.scaler_close = MinMaxScaler(feature_range=(0, 1))
        self.scaler_volume = MinMaxScaler(feature_range=(0, 1))
        self.scaler_open = MinMaxScaler(feature_range=(0, 1))
        self.scaler_high = MinMaxScaler(feature_range=(0, 1))
        self.scaler_low = MinMaxScaler(feature_range=(0, 1))
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
        self.df = pd.DataFrame(data[["Close", "Volume", "Open", "High", "Low"]])

    def preprocess_data(self):
        self.df_scaled = self.scaler_close.fit_transform(self.df[["Close"]])
        self.scaler_volume.fit(self.df[["Volume"]])
        self.scaler_open.fit(self.df[["Open"]])
        self.scaler_high.fit(self.df[["High"]])
        self.scaler_low.fit(self.df[["Low"]])

        train_size = int(len(self.df_scaled) * 0.80)

        train_data_close = self.df_scaled[:train_size]
        train_data_volume = self.scaler_volume.transform(
            self.df[["Volume"]][:train_size]
        )
        train_data_open = self.scaler_open.transform(self.df[["Open"]][:train_size])
        train_data_high = self.scaler_high.transform(self.df[["High"]][:train_size])
        train_data_low = self.scaler_low.transform(self.df[["Low"]][:train_size])
        train_data = np.concatenate(
            (
                train_data_close,
                train_data_volume,
                train_data_open,
                train_data_high,
                train_data_low,
            ),
            axis=1,
        )

        # testing
        test_data_close = self.df_scaled[train_size - self.sequence_length :]
        test_data_volume = self.scaler_volume.transform(
            self.df[["Volume"]][train_size - self.sequence_length :]
        )
        test_data_open = self.scaler_open.transform(
            self.df[["Open"]][train_size - self.sequence_length :]
        )
        test_data_high = self.scaler_high.transform(
            self.df[["High"]][train_size - self.sequence_length :]
        )
        test_data_low = self.scaler_low.transform(
            self.df[["Low"]][train_size - self.sequence_length :]
        )

        test_data = np.concatenate(
            (
                test_data_close,
                test_data_volume,
                test_data_open,
                test_data_high,
                test_data_low,
            ),
            axis=1,
        )
        self.X_train, self.y_train = self.create_sequences(
            train_data, self.sequence_length
        )
        self.X_test, self.y_test = self.create_sequences(
            test_data, self.sequence_length
        )

    def create_sequences(self, data, seq_length):
        sequences, targets = [], []
        for i in range(len(data) - seq_length):
            seq = torch.tensor(data[i : i + seq_length], dtype=torch.float32)
            target = torch.tensor(
                data[i + seq_length : i + seq_length + 1, 0], dtype=torch.float32
            )  # close price is the only target
            sequences.append(seq)
            targets.append(target)
        return torch.stack(sequences), torch.stack(targets).view(-1)

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
                y_pred = self.model(seq.unsqueeze(1))

                # last closing prices
                predicted_close = y_pred[-1][0]

                true_close = labels

                # Calculate loss using only closing prices
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
            predicted_close = y_pred[-1, 0]  # last prediction for close price
            predicted_volume = y_pred[-1, 1]  # last prediction for volume
            self.train_predictions.append(predicted_close.item())

        self.test_predictions = []
        for seq in self.X_test:
            with torch.no_grad():
                self.model.hidden = (
                    torch.zeros(1, 1, self.model.hidden_size),
                    torch.zeros(1, 1, self.model.hidden_size),
                )
            y_pred = self.model(seq.unsqueeze(1))
            predicted_close = y_pred[-1, 0]  # last prediction for close price
            predicted_volume = y_pred[-1, 1]  # last prediction for volume (unused atm)
            self.test_predictions.append(predicted_close.item())

        self.train_predictions = self.scaler_close.inverse_transform(
            np.array(self.train_predictions).reshape(-1, 1)
        )
        self.y_train_inv = self.scaler_close.inverse_transform(
            self.y_train.numpy().reshape(-1, 1)
        )
        self.test_predictions = self.scaler_close.inverse_transform(
            np.array(self.test_predictions).reshape(-1, 1)
        )
        self.y_test_inv = self.scaler_close.inverse_transform(
            self.y_test.numpy().reshape(-1, 1)
        )

        train_rmse = np.sqrt(
            mean_squared_error(self.y_train_inv, self.train_predictions)
        )
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
                # current_sequence = torch.cat(
                #     (current_sequence[1:], next_pred.view(1, 1)), dim=0
                # )

        future_predictions = self.scaler_close.inverse_transform(
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
        # h_n, c_n = encoder_hidden
        # print("Seq2Seq input:", input_seq[:, -1].unsqueeze(0).size())
        # print("Seq2Seq Hidden: ", h_n.size(), c_n.size())
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
        epochs=400,
        interval="1d",
    )

    stock_predictor.download_data()
    stock_predictor.preprocess_data()
    stock_predictor.build_model()
    stock_predictor.train_model(verbose=True)
    train_rmse, test_rmse = stock_predictor.evaluate_model()
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    stock_predictor.plot_predictions()

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
