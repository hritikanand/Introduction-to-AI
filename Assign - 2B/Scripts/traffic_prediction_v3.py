import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import joblib

# Dataset class
class TrafficDataset(Dataset):
    def __init__(self, data, seq_len, target_scat=None):
        self.seq_len = seq_len
        self.data = data.values.astype(np.float32)
        self.target_idx = data.columns.get_loc('Volume')
        self.scat_idx = data.columns.get_loc('SCATS_ID') if 'SCATS_ID' in data.columns else None
        self.target_scat = target_scat
        self.samples = []
        for i in range(len(data) - seq_len - 1):
            if self.target_scat is None or self.data[i+seq_len, self.scat_idx] == self.target_scat:
                x = self.data[i:i+seq_len]
                y = self.data[i+seq_len, self.target_idx]
                self.samples.append((x, y))
    def __len__(self): 
        return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1])
        out = self.relu(self.fc1(out))
        return self.fc2(out)

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1])
        out = self.relu(self.fc1(out))
        return self.fc2(out)

# Combined MSE + MAE loss with Huber loss
class CombinedLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.huber = nn.HuberLoss(delta=delta)
    def forward(self, preds, targets):
        return 0.4 * self.mse(preds, targets) + 0.3 * self.mae(preds, targets) + 0.3 * self.huber(preds, targets)

# Training function with patience-based early stopping
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, model_name="model", patience=10):
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x).squeeze()
                loss = criterion(pred, y)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    model.load_state_dict(best_model_state)
    save_path = f"{model_name}_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to: {os.path.abspath(save_path)}")
    return save_path

# Evaluation function
def evaluate(model, loader, scaler=None):
    model.eval()
    flows, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).squeeze().detach().cpu().numpy()
            flows.extend(preds)
            targets.extend(y.cpu().numpy())
    flows = np.array(flows)
    targets = np.array(targets)
    if scaler:
        flows = scaler.inverse_transform(flows.reshape(-1,1)).flatten()
        targets = scaler.inverse_transform(targets.reshape(-1,1)).flatten()
    return flows, targets

# Random Forest model training
def train_random_forest(data, seq_len, target_scat):
    X, y = [], []
    for i in range(len(data) - seq_len - 1):
        if data.iloc[i+seq_len]['SCATS_ID'] == target_scat:
            x_seq = data.iloc[i:i+seq_len].values.flatten()
            X.append(x_seq)
            y.append(data.iloc[i+seq_len]['Volume'])
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def evaluate_random_forest(model, data, seq_len, target_scat, scaler=None):
    X, y_true = [], []
    for i in range(len(data) - seq_len - 1):
        if data.iloc[i+seq_len]['SCATS_ID'] == target_scat:
            x_seq = data.iloc[i:i+seq_len].values.flatten()
            X.append(x_seq)
            y_true.append(data.iloc[i+seq_len]['Volume'])
    preds = model.predict(X)
    if scaler:
        preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
        y_true = scaler.inverse_transform(np.array(y_true).reshape(-1,1)).flatten()
    return preds, y_true

# Export results to csv
def save_all_predictions(true, lstm_pred, gru_pred, rf_pred, filename="all_model_predictions.csv"):
    df = pd.DataFrame({
        "Actual traffic volume": true,
        "LSTM_Predicted": lstm_pred,
        "GRU_Predicted": gru_pred,
        "RF_Predicted": rf_pred
    })
    df.to_csv(filename, index=False)
    print(f"All predictions saved to: {os.path.abspath(filename)}")

# MAIN
if __name__ == "__main__":
    df = pd.read_csv("SCATS_timeseries.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    df = df.sort_values('Timestamp')
    df['hour'] = df['Timestamp'].dt.hour
    df['minute'] = df['Timestamp'].dt.minute

    # Data check
    features = ['SCATS_ID', 'hour', 'minute', 'Volume']
    df = df[features]
    assert not df.empty, "DataFrame is still empty after cleaning. Check CSV format."

    # Parameters
    SEQ_LEN = 24
    BATCH_SIZE = 64
    HIDDEN_SIZE = 192
    EPOCHS = 100
    TARGET_SCAT = 970

    # Normalise all numerical features
    scaler = MinMaxScaler()
    df['Volume'] = scaler.fit_transform(df[['Volume']])

    # Split data
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]

    # Dataset and dataloader
    train_ds = TrafficDataset(train_df, SEQ_LEN, TARGET_SCAT)
    val_ds = TrafficDataset(val_df, SEQ_LEN, TARGET_SCAT)
    test_ds = TrafficDataset(test_df, SEQ_LEN, TARGET_SCAT)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_size = train_df.shape[1]
    output_size = 1

    lstm_model = LSTMModel(input_size, HIDDEN_SIZE, num_layers=2, output_size=output_size, dropout=0.2)
    gru_model = GRUModel(input_size, HIDDEN_SIZE, num_layers=2, output_size=output_size, dropout=0.2)
    criterion = CombinedLoss(delta=0.8)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.0008, weight_decay=1e-6)
    gru_optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.0008, weight_decay=1e-6)
    lstm_scheduler = ReduceLROnPlateau(lstm_optimizer, mode='min', factor=0.8, patience=8)
    gru_scheduler = ReduceLROnPlateau(gru_optimizer, mode='min', factor=0.8, patience=8)

    print("Training LSTM...")
    lstm_pth = train(lstm_model, train_loader, val_loader, criterion, lstm_optimizer, lstm_scheduler, EPOCHS, model_name="lstm", patience=10)

    print("\nTraining GRU...")
    gru_pth = train(gru_model, train_loader, val_loader, criterion, gru_optimizer, gru_scheduler, EPOCHS, model_name="gru", patience=10)

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(train_df, SEQ_LEN, TARGET_SCAT)
    rf_model_path = "random_forest_best.pkl"
    joblib.dump(rf_model, rf_model_path)
    print(f"Random Forest model saved to: {os.path.abspath(rf_model_path)}")

    print("\nEvaluating LSTM...")
    lstm_flow, lstm_targets = evaluate(lstm_model, test_loader, scaler)

    print("\nEvaluating GRU...")
    gru_flow, gru_targets = evaluate(gru_model, test_loader, scaler)

    print("\nEvaluating Random Forest...")
    rf_flow, rf_targets = evaluate_random_forest(rf_model, test_df, SEQ_LEN, TARGET_SCAT, scaler)

    # Save all results to csv file
    min_len = min(len(lstm_targets), len(gru_targets), len(rf_targets))
    save_all_predictions(
        lstm_targets[:min_len],
        lstm_flow[:min_len],
        gru_flow[:min_len],
        rf_flow[:min_len],
        filename="all_model_predictions.csv"
    )
