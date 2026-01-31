#!/usr/bin/env python3
"""
AI Trading Pipeline v2.1 - CONTINUOUS LEARNING
Features:
1. Model persistence (save/load)
2. Incremental training
3. Performance tracking
4. Gradual improvement
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import MetaTrader5 as mt5
from supabase import create_client
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
MODEL_PATH = "trading_model_v2.h5"
SCALER_PATH = "scaler_v2.pkl"
METRICS_PATH = "model_metrics.json"

# Supabase Configuration
SUPABASE_URL = "https://kktfrmvwzykkzosvzddn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtrdGZybXZ3enlra3pvc3Z6ZGRuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg3MTczNzAsImV4cCI6MjA4NDI5MzM3MH0.Cr16JZvwQx_MrW-Gh-S-p_mLTCTnMMqTFMJk40J8F5g"

# MT5 Configuration
MT5_LOGIN = 104492090
MT5_PASSWORD = "Ug.9vEPg"
MT5_SERVER = "FBSMarkets-Demo"

# Model Configuration
MODEL_VERSION = "v2.1.0"
PREDICTION_THRESHOLD = 0.70
RETRAIN_EPOCHS = 20  # Fewer epochs for fine-tuning
FULL_TRAIN_EPOCHS = 100  # Full training from scratch
BATCH_SIZE = 32

# Trading Pairs
SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD',
    'USDCAD', 'USDCHF', 'EURJPY', 'GBPJPY', 'EURGBP', 'XAUUSD'
]

# ============================================================================
# SUPABASE CLIENT
# ============================================================================

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_model_and_scaler(model, scaler):
    """Save model and scaler to disk"""
    model.save(MODEL_PATH)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Model saved to {MODEL_PATH}")

def load_model_and_scaler():
    """Load existing model and scaler"""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = load_model(MODEL_PATH)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"📂 Loaded existing model from {MODEL_PATH}")
        return model, scaler
    return None, None

# ============================================================================
# MT5 CONNECTION
# ============================================================================

def connect_mt5():
    """Connect to MetaTrader 5"""
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return False
    
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        print(f"❌ MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    
    print(f"✅ Connected to MT5: {mt5.account_info().login}")
    return True

# ============================================================================
# DATA FETCHING & FEATURES (same as v2.0)
# ============================================================================

def fetch_ohlc_data(symbol, timeframe=mt5.TIMEFRAME_H1, periods=500):
    """Fetch OHLC data from MT5"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, periods)
    
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_features(df):
    """Calculate technical indicators"""
    # (Same as v2.0 - full feature engineering code)
    # Price changes
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving Averages
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    df['ema_5'] = df['close'].ewm(span=5).mean()
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    
    # Crossovers
    df['sma_cross_5_10'] = (df['sma_5'] - df['sma_10']) / df['close']
    df['sma_cross_10_20'] = (df['sma_10'] - df['sma_20']) / df['close']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_normalized'] = (df['rsi'] - 50) / 50
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_normalized'] = df['macd_hist'] / df['close']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['atr_normalized'] = df['atr'] / df['close']
    
    # Momentum
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    return df

def create_labels(df, forward_periods=5, threshold=0.001):
    """Create labels"""
    df['future_price'] = df['close'].shift(-forward_periods)
    df['price_change'] = (df['future_price'] - df['close']) / df['close']
    
    df['label'] = 1  # HOLD
    df.loc[df['price_change'] > threshold, 'label'] = 2  # BUY
    df.loc[df['price_change'] < -threshold, 'label'] = 0  # SELL
    
    return df

# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_improved_model(input_shape):
    """Build improved model"""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,),
              kernel_regularizer=keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu',
              kernel_regularizer=keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu',
              kernel_regularizer=keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu',
              kernel_regularizer=keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================================================
# TRAINING WITH PERSISTENCE
# ============================================================================

def train_or_finetune(X, y, existing_model=None):
    """Train new model or fine-tune existing one"""
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset: {len(X_train)} train, {len(X_val)} validation")
    
    # Class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Decide: new model or fine-tune
    if existing_model is not None:
        print("🔄 Fine-tuning existing model...")
        model = existing_model
        epochs = RETRAIN_EPOCHS
        learning_rate = 0.0001  # Lower LR for fine-tuning
        
        # Update learning rate
        keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
    else:
        print("🆕 Training new model from scratch...")
        model = build_improved_model(X_train.shape[1])
        epochs = FULL_TRAIN_EPOCHS
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15 if existing_model is None else 5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5 if existing_model is None else 3,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\n📈 Performance:")
    print(f"   Training: {train_acc*100:.2f}%")
    print(f"   Validation: {val_acc*100:.2f}%")
    print(f"   Gap: {(train_acc - val_acc)*100:.2f}%")
    
    if train_acc - val_acc > 0.15:
        print(f"   ⚠️ Overfitting detected!")
    
    return model

# ============================================================================
# PREDICTION
# ============================================================================

def make_predictions(model, scaler, symbol, df):
    """Make predictions"""
    feature_cols = [col for col in df.columns if col not in 
                    ['time', 'label', 'future_price', 'price_change', 
                     'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume',
                     'tr1', 'tr2', 'tr3', 'tr']]
    
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled, verbose=0)
    latest_pred = predictions[-1]
    pred_class = np.argmax(latest_pred)
    confidence = latest_pred[pred_class]
    
    signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    signal = signal_map[pred_class]
    
    if confidence >= PREDICTION_THRESHOLD and signal != 'HOLD':
        now_utc = datetime.now(timezone.utc)
        timestamp_unix = int(now_utc.timestamp())
        
        data = {
            'timestamp': now_utc.isoformat(),
            'timestamp_unix': timestamp_unix,
            'symbol': symbol,
            'prediction': signal,
            'confidence': float(confidence),
            'model_version': MODEL_VERSION
        }
        
        try:
            supabase.table('predictions').insert(data).execute()
            print(f"   ✅ {symbol}: {signal} ({confidence*100:.1f}%)")
            return True
        except Exception as e:
            print(f"   ❌ {symbol}: {e}")
            return False
    else:
        print(f"   ⏭️ {symbol}: Skipped ({confidence*100:.1f}%)")
        return False

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution with continuous learning"""
    
    print("="*60)
    print("🤖 AI Trading Pipeline v2.1 - CONTINUOUS LEARNING")
    print("="*60)
    
    # Connect MT5
    if not connect_mt5():
        return
    
    # Try load existing model
    existing_model, existing_scaler = load_model_and_scaler()
    
    # Collect data
    print("\n📥 Fetching market data...")
    all_data = []
    
    for symbol in SYMBOLS:
        df = fetch_ohlc_data(symbol, periods=500)
        if df is not None:
            df = calculate_features(df)
            df = create_labels(df, threshold=0.001)
            df = df.dropna()
            
            if len(df) > 0:
                all_data.append(df)
                print(f"   ✅ {symbol}: {len(df)} samples")
    
    if not all_data:
        print("❌ No data!")
        mt5.shutdown()
        return
    
    # Combine
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Total: {len(combined_df)} samples")
    
    # Prepare features
    feature_cols = [col for col in combined_df.columns if col not in 
                    ['time', 'label', 'future_price', 'price_change',
                     'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume',
                     'tr1', 'tr2', 'tr3', 'tr']]
    
    X = combined_df[feature_cols].values
    y = combined_df['label'].values
    
    # Scale
    if existing_scaler is not None:
        scaler = existing_scaler
        X_scaled = scaler.transform(X)
        print("📐 Using existing scaler")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("📐 Created new scaler")
    
    # Train or fine-tune
    model = train_or_finetune(X_scaled, y, existing_model)
    
    # Save model & scaler
    save_model_and_scaler(model, scaler)
    
    # Make predictions
    print(f"\n🔮 Making predictions (threshold: {PREDICTION_THRESHOLD*100}%)...")
    saved_count = 0
    
    for symbol in SYMBOLS:
        df = fetch_ohlc_data(symbol, periods=100)
        if df is not None:
            df = calculate_features(df)
            df = df.dropna()
            
            if len(df) > 0:
                if make_predictions(model, scaler, symbol, df):
                    saved_count += 1
    
    print(f"\n✅ Saved {saved_count} predictions")
    
    mt5.shutdown()
    print("\n🏁 Complete!")
    print(f"💾 Model will be reused in next run for faster training")

if __name__ == "__main__":
    main()
