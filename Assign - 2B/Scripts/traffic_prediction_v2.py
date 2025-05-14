import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

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
        return torch.tensor(x), torch.tensor(y)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
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
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
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

# Training function
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    model.train()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        start_time = time.time()
        total_train_loss = 0
        
        # Training phase
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x).squeeze()
                loss = criterion(pred, y)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduler step
        scheduler.step(avg_val_loss)
        
        # Saves the best model state internally (optional, can be removed)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            
        time_elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} | Time: {time_elapsed:.2f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # Load the best model state
    model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

# Evaluation function
def evaluate(model, loader, scaler=None, name="Model"):
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

    rmse = np.sqrt(mean_squared_error(targets, flows))
    mae = mean_absolute_error(targets, flows)
    mape = np.mean(np.abs((targets - flows) / (targets + 1e-5))) * 100  # Added MAPE
    
    print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
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

    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    mape = np.mean(np.abs((y_true - preds) / (y_true + 1e-5))) * 100
    
    print(f"Random Forest - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
    return preds, y_true

# Added time features
def add_time_features(df):
    # Extract more time features
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['day_of_month'] = df['Timestamp'].dt.day
    df['month'] = df['Timestamp'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    
    return df

# Plot loss curves
def plot_loss_curves(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

# MAIN
if __name__ == "__main__":
    df = pd.read_csv("SCATS_timeseries.csv")

    # Parse timestamps
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Timestamp'])

    df = df.sort_values('Timestamp')
    df['hour'] = df['Timestamp'].dt.hour
    df['minute'] = df['Timestamp'].dt.minute
    
    # Added more time features for better prediction
    df = add_time_features(df)

    features = ['SCATS_ID', 'hour', 'minute', 'day_of_week', 'is_weekend', 
                'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'Volume']
    df = df[features]

    # Check if df is empty
    assert not df.empty, "DataFrame is still empty after cleaning. Check CSV format."

    # Parameters
    SEQ_LEN = 24  # Increased for better context
    BATCH_SIZE = 64
    HIDDEN_SIZE = 192  # Increased for more representation capacity
    EPOCHS = 100 
    TARGET_SCAT = 970

    # Normalise all numerical features
    scaler = MinMaxScaler()
    volume_idx = df.columns.get_loc('Volume')
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

    # Improved models
    lstm_model = LSTMModel(input_size, HIDDEN_SIZE, num_layers=2, output_size=output_size, dropout=0.2)
    gru_model = GRUModel(input_size, HIDDEN_SIZE, num_layers=2, output_size=output_size, dropout=0.2)

    # Enhanced loss function
    criterion = CombinedLoss(delta=0.8)
    
    # Optimisers with weight decay for regularisation - adjusted learning rate and weight decay
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.0008, weight_decay=1e-6)
    gru_optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.0008, weight_decay=1e-6)
    
    # Learning rate schedulers - more gradual learning rate reduction
    lstm_scheduler = ReduceLROnPlateau(lstm_optimizer, mode='min', factor=0.8, patience=8)
    gru_scheduler = ReduceLROnPlateau(gru_optimizer, mode='min', factor=0.8, patience=8)

    print("Training LSTM...")
    lstm_train_losses, lstm_val_losses = train(lstm_model, train_loader, val_loader, criterion, lstm_optimizer, lstm_scheduler, EPOCHS)

    print("\nTraining GRU...")
    gru_train_losses, gru_val_losses = train(gru_model, train_loader, val_loader, criterion, gru_optimizer, gru_scheduler, EPOCHS)

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(train_df, SEQ_LEN, TARGET_SCAT)

    print("\nEvaluating LSTM...")
    lstm_flow, lstm_targets = evaluate(lstm_model, test_loader, scaler, "LSTM")

    print("\nEvaluating GRU...")
    gru_flow, gru_targets = evaluate(gru_model, test_loader, scaler, "GRU")

    print("\nEvaluating Random Forest...")
    rf_flow, rf_targets = evaluate_random_forest(rf_model, test_df, SEQ_LEN, TARGET_SCAT, scaler)

    # Visualisation
    predictions = {
        'LSTM': lstm_flow,
        'GRU': gru_flow,
        'Random Forest': rf_flow
    }
    
    # Creates a simple plot with plt.show()
    plt.figure(figsize=(15, 8))
    plt.plot(lstm_targets[:100], 'k-', label='True Values', linewidth=2)
    plt.plot(lstm_flow[:100], 'b-', label='LSTM Predicted')
    plt.plot(gru_flow[:100], 'r-', label='GRU Predicted')
    plt.plot(rf_flow[:100], 'g-', label='RF Predicted')
    plt.title(f"Traffic Flow Prediction for SCATS_ID {TARGET_SCAT}")
    plt.xlabel("Time (15-min intervals)")
    plt.ylabel("Traffic Volume")
    plt.legend()
    plt.show()
    
    # Compare model performance
    print("\nModel Comparison Summary:")
    print("=" * 50)
    for name, preds in predictions.items():
        rmse = np.sqrt(mean_squared_error(lstm_targets, preds))
        mae = mean_absolute_error(lstm_targets, preds)
        mape = np.mean(np.abs((lstm_targets - preds) / (lstm_targets + 1e-5))) * 100
        print(f"{name:15s}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, MAPE = {mape:.2f}%")