import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import joblib # For saving the scaler
import os
import re # For sanitizing filenames

# --- Configuration ---
DATA_PATH = "Chiaki_wine_prediction/Top37.csv" # Path to your data file
MODEL_SAVE_DIR = "saved_models" # Directory to save models and scalers
LAG = 12  # Number of past observations to use for prediction (must match Streamlit app)
EPOCHS = 50 # Number of training epochs
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 10 # Increased patience slightly

# --- Helper Function for Filenames ---
def sanitize_filename(name):
    """Removes characters problematic for filenames and shortens."""
    # Remove or replace special characters
    name = re.sub(r'[,\'&]', '', name)
    name = re.sub(r'\s+', '_', name) # Replace spaces with underscores
    # Limit length to avoid issues on some file systems
    return name[:50]

# --- Data Loading ---
try:
    df = pd.read_csv(DATA_PATH, delimiter=",")
    # Convert DateTime column to date format
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime']) # Drop rows where date couldn't be parsed
    df = df.set_index('Datetime')

    # Convert all other columns to numeric format
    wine_columns = df.columns.tolist()
    for col in wine_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Forward fill missing values AFTER numeric conversion
    df = df.ffill()
    # Handle potential remaining NaNs at the beginning
    df = df.bfill()

    if df.isna().sum().sum() > 0:
         print("Warning: NaN values remain after ffill and bfill. Check input data.")
         print(df.isna().sum())
         # Consider stopping or handling differently if NaNs persist

    # Calculate percentage returns (monthly)
    returns = df.pct_change().dropna()
    wine_columns = returns.columns.tolist() # Update wine_columns based on returns df

except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()
except Exception as e:
    print(f"Error loading or processing data: {e}")
    exit()

# --- Ensure Save Directory Exists ---
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
    print(f"Created directory: {MODEL_SAVE_DIR}")

# --- Model Training and Saving Loop ---
print("Starting model training and saving process...")

for wine in wine_columns:
    print(f"\n--- Processing: {wine} ---")
    time_series_pd = returns[wine].copy() # Work with the pandas Series

    if len(time_series_pd) < LAG + 20: # Check if enough data for lag, train, test split logic
         print(f"Skipping {wine}: Insufficient data points ({len(time_series_pd)})")
         continue

    # 1. Prepare Data for LSTM
    scaler = StandardScaler()
    # Fit on the entire series for this wine to ensure consistent scaling
    time_series_scaled = scaler.fit_transform(time_series_pd.values.reshape(-1, 1))

    # 2. Create Lagged Features
    def create_lagged_features(data, lag):
        X, y = [], []
        # Start loop from lag index to ensure we have enough past data
        for i in range(lag, len(data)):
             # Extract sequence from i-lag to i (exclusive of i)
             sequence = data[i-lag:i]
             # The target is the value at index i
             target = data[i]
             X.append(sequence)
             y.append(target)
        # Ensure X and y are numpy arrays of the correct shape
        # X should be (num_samples, lag, 1 feature)
        # y should be (num_samples, 1 feature)
        return np.array(X).reshape(-1, lag, 1), np.array(y)


    X, y = create_lagged_features(time_series_scaled, LAG)

    if X.shape[0] == 0:
        print(f"Skipping {wine}: Not enough data after creating lagged features.")
        continue

    # 3. Split into training and testing sets (Optional here, but good practice to verify)
    # We will train the final model on *all* available lagged data (X, y)
    # train_size = int(0.80 * len(X))
    # X_train, X_test = X[:train_size], X[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]
    X_train, y_train = X, y # Use all data for the final saved model

    # 4. Build LSTM Model
    # Clear previous session if running in interactive environment (like Jupyter)
    tf.keras.backend.clear_session()

    model = tf.keras.Sequential([
        # Use Input layer to define shape explicitly
        tf.keras.layers.Input(shape=(LAG, 1)),
        LSTM(50, activation='relu', return_sequences=True),
        LSTM(50, activation='relu'), # Second LSTM layer
        tf.keras.layers.Dense(1) # Output layer
    ])
    model.compile(optimizer='adam', loss='mse')
    # model.summary() # Optional: print model summary

    # 5. Define Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss',         # Monitor validation loss
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True, # Restore weights from the epoch with the best val_loss
        verbose=1
    )

    # 6. Fit the Model (using a validation split from the training data)
    print(f"Training model for {wine}...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT, # Use part of training data for validation
        callbacks=[early_stop],
        verbose=0 # Set to 1 or 2 for more output during training
    )
    print(f"Training finished for {wine}. Best validation loss: {min(history.history.get('val_loss', [np.inf])):.4f}")


    # 7. Save the Model and Scaler
    try:
        sanitized_name = sanitize_filename(wine)
        model_path = os.path.join(MODEL_SAVE_DIR, f"model_{sanitized_name}.keras")
        scaler_path = os.path.join(MODEL_SAVE_DIR, f"scaler_{sanitized_name}.joblib")

        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        print(f"Successfully saved model to: {model_path}")
        print(f"Successfully saved scaler to: {scaler_path}")

    except Exception as e:
        print(f"!!! Error saving model or scaler for {wine}: {e}")

print("\n--- Model training and saving process complete. ---")