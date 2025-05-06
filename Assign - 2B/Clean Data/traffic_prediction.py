import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Dataset class
class TrafficDataset(Dataset):
    def __init__(self, data, seq_len, target_sensor=None):
        self.seq_len = seq_len
        self.data = data.values.astype(np.float32)
        self.target_idx = data.columns.get_loc(target_sensor) if target_sensor else None
        self.samples = []
        for i in range(len(data) - seq_len):
            x = self.data[i:i+seq_len]
            y = self.data[i+seq_len, self.target_idx] if self.target_idx is not None else self.data[i+seq_len]
            self.samples.append((x, y))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

# Training function
def train(model, loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

# Evaluation function
def evaluate(model, loader, scaler=None):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).squeeze().item()
            preds.append(pred)
            targets.append(y.item())
    if len(preds) == 0:
        print("Warning: No predictions to inverse transform!")
        return [], []
    if scaler:
        preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
        targets = scaler.inverse_transform(np.array(targets).reshape(-1,1)).flatten()
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return preds, targets

if __name__ == "__main__":
    df = pd.read_csv("SCATS_timeseries.csv", parse_dates=['Timestamp'], index_col='Timestamp')
    print(df.head())
    print(df.shape)
    SEQ_LEN = 16  # 12 hours * 4 intervals/hour (15-min intervals) (should be 48, changing it for testing)
    BATCH_SIZE = 32
    HIDDEN_SIZE = 128
    EPOCHS = 30
    TARGET_SENSOR = '0970'  # Change as needed

    # Scale only target sensor
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[TARGET_SENSOR] = scaler.fit_transform(df[[TARGET_SENSOR]])

    # adjusts train/ test split
    train_size = int(len(df) * 0.9)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    train_df = df.iloc[:int(len(df)*0.8)]
    test_df = df.iloc[int(len(df)*0.8):]

    # for debugging purposes
    print(f"Total data points: {len(df)}") 
    print(f"Train data points: {len(train_df)}")
    print(f"Test data points: {len(test_df)}")
    print(f"SEQ_LEN: {SEQ_LEN}")
    print(f"Possible test samples: {len(test_df) - SEQ_LEN}")
    print(f"Target sensor '{TARGET_SENSOR}' in columns? {TARGET_SENSOR in df.columns}")

    if len(test_df) <= SEQ_LEN:
        print("Warning: Test set too small for the chosen SEQ_LEN. Consider reducing SEQ_LEN or increasing test set size.")

    # Create datasets
    train_ds = TrafficDataset(train_df, SEQ_LEN, TARGET_SENSOR)
    test_ds = TrafficDataset(test_df, SEQ_LEN, TARGET_SENSOR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    input_size = len(df.columns)
    output_size = 1

    # Train and evaluate LSTM
    print("Training LSTM...")
    lstm_model = LSTMModel(input_size, HIDDEN_SIZE, output_size)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train(lstm_model, train_loader, criterion, lstm_optimizer, EPOCHS)
    print("Evaluating LSTM...")
    lstm_preds, lstm_targets = evaluate(lstm_model, test_loader, scaler)

    # Train and evaluate GRU
    print("Training GRU...")
    gru_model = GRUModel(input_size, HIDDEN_SIZE, output_size)
    gru_optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)
    train(gru_model, train_loader, criterion, gru_optimizer, EPOCHS)
    print("Evaluating GRU...")
    gru_preds, gru_targets = evaluate(gru_model, test_loader, scaler)

    # Plot results
    plt.figure(figsize=(15,6))
    plt.plot(lstm_targets, label='True')
    plt.plot(lstm_preds, label='LSTM Prediction')
    plt.plot(gru_preds, label='GRU Prediction')
    plt.title(f"Traffic Flow Prediction for SCAT Number {TARGET_SENSOR}")
    plt.xlabel("Time steps (15-min intervals)")
    plt.ylabel("Traffic Flow (vehicles per 15min)")
    plt.legend()
    plt.show()