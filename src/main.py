"""
AI Trading System - Complete Python Pipeline
Reads data from Supabase → Trains AI → Writes predictions back
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Supabase
from supabase import create_client, Client

# ================================================================
# CONFIGURATION
# ================================================================

class Config:
    import os

class Config:
    # Supabase credentials - baca dari environment variable
    SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://kktfrmvwzykkzosvzddn.supabase.co')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
    
    # Model settings
    MODEL_VERSION = "v1.0.0"
    LOOKBACK_DAYS = 30  # How many days of historical data to use
    
    # Trading logic
    PREDICTION_THRESHOLD = 0.65  # Minimum confidence for signal
    SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
    TIMEFRAME = 'H1'
    
   # Feature engineering - REDUCED untuk data yang masih sedikit
    MOMENTUM_PERIODS = [3, 5, 10]  # Dari [5,10,20] jadi [3,5,10]
    VOLATILITY_WINDOW = 10  # Dari 20 jadi 10
    
    # Training
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

# ================================================================
# 1. DATA LOADER - Read from Supabase
# ================================================================

class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    
    def load_ohlc_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Load OHLC data from Supabase"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        try:
            response = self.supabase.table('market_data_ohlc')\
                .select('*')\
                .eq('symbol', symbol)\
                .eq('timeframe', self.config.TIMEFRAME)\
                .gte('timestamp', start_date.isoformat())\
                .order('timestamp', desc=False)\
                .execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                print(f"✅ Loaded {len(df)} OHLC records for {symbol}")
                return df
            else:
                print(f"⚠️ No OHLC data found for {symbol}")
                return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading OHLC: {e}")
            return pd.DataFrame()
    
    def load_indicators(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Load indicators from Supabase"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        try:
            response = self.supabase.table('indicators')\
                .select('*')\
                .eq('symbol', symbol)\
                .eq('timeframe', self.config.TIMEFRAME)\
                .gte('timestamp', start_date.isoformat())\
                .order('timestamp', desc=False)\
                .execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                print(f"✅ Loaded {len(df)} indicator records for {symbol}")
                return df
            else:
                print(f"⚠️ No indicator data found for {symbol}")
                return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading indicators: {e}")
            return pd.DataFrame()

# ================================================================
# 2. FEATURE ENGINEER - Create ML Features
# ================================================================

class FeatureEngineer:
    def __init__(self, config: Config):
        self.config = config
    
    def merge_data(self, ohlc: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
        """Merge OHLC and indicators on timestamp"""
        if ohlc.empty or indicators.empty:
            return pd.DataFrame()
        
        # Merge on timestamp
        df = pd.merge(ohlc, indicators, on=['timestamp', 'symbol', 'timeframe'], how='inner')
        print(f"✅ Merged dataset: {len(df)} records")
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer ML features from raw data"""
        if df.empty:
            return df
        
        print("🔧 Engineering features...")
        
        # Price momentum features
        for period in self.config.MOMENTUM_PERIODS:
            df[f'price_momentum_{period}'] = df['close'].pct_change(period) * 100
        
        # Volatility
        df['volatility_ratio'] = df['atr_14'] / df['close']
        
        # Trend strength (using MACD)
        df['trend_strength'] = abs(df['macd_main'] - df['macd_signal'])
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
        
        # Time-based features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Trading session (simplified)
        df['session'] = df['hour_of_day'].apply(self._get_session)
        df['session_encoded'] = df['session'].map({'asian': 0, 'european': 1, 'us': 2, 'other': 3})
        
        # Target variable: Future return
        df['future_return_1h'] = df['close'].shift(-1) / df['close'] - 1
        df['future_return_4h'] = df['close'].shift(-4) / df['close'] - 1
        
        # Binary classification targets
        df['profitable_long'] = (df['future_return_1h'] > 0.0002).astype(int)  # >2 pips
        df['profitable_short'] = (df['future_return_1h'] < -0.0002).astype(int)
        
        # Drop rows with NaN
        df = df.dropna()
        
        print(f"✅ Features created: {len(df)} complete records")
        return df
    
    def _get_session(self, hour: int) -> str:
        """Determine trading session based on hour (UTC)"""
        if 0 <= hour < 7:
            return 'asian'
        elif 7 <= hour < 15:
            return 'european'
        elif 15 <= hour < 21:
            return 'us'
        else:
            return 'other'
    
    def save_features_to_db(self, df: pd.DataFrame, supabase: Client):
        """Save engineered features to ml_features table"""
        if df.empty:
            return
        
        print("💾 Saving features to database...")
        
        records = []
        for _, row in df.iterrows():
            record = {
                'timestamp': row['timestamp'].isoformat(),
                'symbol': row['symbol'],
                'timeframe': row['timeframe'],
                'price_momentum_5': float(row.get('price_momentum_5', 0)),
                'price_momentum_20': float(row.get('price_momentum_20', 0)),
                'volatility_ratio': float(row.get('volatility_ratio', 0)),
                'trend_strength': float(row.get('trend_strength', 0)),
                'volume_ratio': float(row.get('volume_ratio', 0)),
                'hour_of_day': int(row['hour_of_day']),
                'day_of_week': int(row['day_of_week']),
                'session': row['session'],
                'future_return_1h': float(row.get('future_return_1h', 0)),
                'future_return_4h': float(row.get('future_return_4h', 0)),
                'profitable_long': bool(row.get('profitable_long', False)),
                'profitable_short': bool(row.get('profitable_short', False))
            }
            records.append(record)
        
        try:
            # Insert in batches
            batch_size = 100
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                supabase.table('ml_features').upsert(batch).execute()
            
            print(f"✅ Saved {len(records)} feature records to database")
        except Exception as e:
            print(f"❌ Error saving features: {e}")

# ================================================================
# 3. AI MODEL - Train and Predict
# ================================================================

class AIModel:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.metrics = {}
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for training"""
        # Select features
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi_14', 'macd_main', 'macd_signal', 
            'bb_upper', 'bb_middle', 'bb_lower', 'atr_14',
            'stoch_main', 'stoch_signal',
            'price_momentum_5', 'price_momentum_10', 'price_momentum_20',
            'volatility_ratio', 'trend_strength', 'volume_ratio',
            'hour_of_day', 'day_of_week', 'session_encoded'
        ]
        
        # Filter available columns
        available_cols = [col for col in self.feature_columns if col in df.columns]
        
        X = df[available_cols].values
        
        # Create multi-class target: 0=SELL, 1=HOLD, 2=BUY
        y = np.zeros(len(df))
        y[df['profitable_long'] == 1] = 2  # BUY
        y[df['profitable_short'] == 1] = 0  # SELL
        # Rest remains 1 (HOLD)
        
        self.feature_columns = available_cols
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the AI model"""
        print("🤖 Training AI model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model (Random Forest)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, self.model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, self.model.predict(X_test_scaled))
        
        self.metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_samples': len(y_train),
            'test_samples': len(y_test)
        }
        
        print(f"✅ Model trained!")
        print(f"   Train accuracy: {train_acc:.4f}")
        print(f"   Test accuracy:  {test_acc:.4f}")
        
        return self.model
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence scores"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Get confidence (max probability)
        confidence = np.max(probabilities, axis=1)
        
        return predictions, confidence
    
    def get_signal(self, prediction: int, confidence: float) -> str:
        """Convert prediction to trading signal"""
        if confidence < self.config.PREDICTION_THRESHOLD:
            return 'HOLD'
        
        if prediction == 2:
            return 'BUY'
        elif prediction == 0:
            return 'SELL'
        else:
            return 'HOLD'

# ================================================================
# 4. PREDICTION WRITER - Save predictions to Supabase
# ================================================================

class PredictionWriter:
    def __init__(self, config: Config, supabase: Client):
        self.config = config
        self.supabase = supabase
    
    def save_prediction(self, symbol: str, signal: str, confidence: float):
        """Save single prediction to database"""
        try:
            record = {
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': symbol,
                'model_version': self.config.MODEL_VERSION,
                'prediction': signal,
                'confidence': float(confidence),
                'executed': False
            }
            
            self.supabase.table('predictions').insert(record).execute()
            print(f"✅ Prediction saved: {symbol} → {signal} ({confidence:.2%})")
        except Exception as e:
            print(f"❌ Error saving prediction: {e}")
    
    def save_model_version(self, metrics: Dict):
        """Save model version info"""
        try:
            record = {
                'version': self.config.MODEL_VERSION,
                'model_type': 'RandomForest',
                'train_accuracy': float(metrics.get('train_accuracy', 0)),
                'val_accuracy': 0.0,
                'test_accuracy': float(metrics.get('test_accuracy', 0)),
                'is_active': True
            }
            
            self.supabase.table('model_versions').upsert(record).execute()
            print(f"✅ Model version saved: {self.config.MODEL_VERSION}")
        except Exception as e:
            print(f"❌ Error saving model version: {e}")

# ================================================================
# 5. MAIN PIPELINE
# ================================================================

class TradingPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.loader = DataLoader(config)
        self.engineer = FeatureEngineer(config)
        self.model = AIModel(config)
        self.supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        self.writer = PredictionWriter(config, self.supabase)
    
    def run_training(self, symbol: str):
        """Complete training pipeline for one symbol"""
        print(f"\n{'='*60}")
        print(f"TRAINING PIPELINE: {symbol}")
        print(f"{'='*60}\n")
        
        # 1. Load data
        ohlc = self.loader.load_ohlc_data(symbol, days=self.config.LOOKBACK_DAYS)
        indicators = self.loader.load_indicators(symbol, days=self.config.LOOKBACK_DAYS)
        
        if ohlc.empty or indicators.empty:
            print(f"❌ Insufficient data for {symbol}")
            return None
        
        # 2. Merge and engineer features
        df = self.engineer.merge_data(ohlc, indicators)
        df = self.engineer.create_features(df)
        
        if len(df) < 100:
            print(f"❌ Not enough data after feature engineering: {len(df)} records")
            return None
        
        # 3. Save features to database
        self.engineer.save_features_to_db(df, self.supabase)
        
        # 4. Train model
        X, y = self.model.prepare_training_data(df)
        self.model.train(X, y)
        
        # 5. Save model version
        self.writer.save_model_version(self.model.metrics)
        
        return self.model
    
    def run_prediction(self, symbol: str):
        """Generate prediction for current market state"""
        print(f"\n📊 Generating prediction for {symbol}...")
        
        # Load latest data (last 5 records for feature calculation)
        ohlc = self.loader.load_ohlc_data(symbol, days=2)
        indicators = self.loader.load_indicators(symbol, days=2)
        
        if ohlc.empty or indicators.empty:
            print(f"❌ No recent data for {symbol}")
            return
        
        # Merge and engineer features
        df = self.engineer.merge_data(ohlc, indicators)
        df = self.engineer.create_features(df)
        
        if df.empty:
            print(f"❌ Failed to create features for {symbol}")
            return
        
        # Get latest record
        latest = df.iloc[-1]
        X_latest = latest[self.model.feature_columns].values.reshape(1, -1)
        
        # Predict
        prediction, confidence = self.model.predict(X_latest)
        signal = self.model.get_signal(prediction[0], confidence[0])
        
        # Save to database
        self.writer.save_prediction(symbol, signal, confidence[0])
        
        print(f"   Signal: {signal} | Confidence: {confidence[0]:.2%}")

# ================================================================
# 6. MAIN EXECUTION
# ================================================================

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║          AI TRADING SYSTEM - PYTHON PIPELINE              ║
    ║        Data → Features → Train → Predict → Save          ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    config = Config()
    pipeline = TradingPipeline(config)
    
    # STEP 1: Train models for all symbols
    print("\n🚀 STEP 1: TRAINING MODELS\n")
    
    for symbol in config.SYMBOLS:
        try:
            model = pipeline.run_training(symbol)
            if model:
                print(f"✅ {symbol} training completed\n")
            else:
                print(f"⚠️ {symbol} training skipped\n")
        except Exception as e:
            print(f"❌ Error training {symbol}: {e}\n")
    
    # STEP 2: Generate predictions for all symbols
    print("\n🔮 STEP 2: GENERATING PREDICTIONS\n")
    
    for symbol in config.SYMBOLS:
        try:
            pipeline.run_prediction(symbol)
        except Exception as e:
            print(f"❌ Error predicting {symbol}: {e}")
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED!")
    print("="*60)
    print("\n📌 Next steps:")
    print("   1. Check 'predictions' table in Supabase")
    print("   2. Enable ENABLE_TRADING in your EA")
    print("   3. EA will read predictions and execute trades")
    print("\n")

if __name__ == "__main__":
    main()
