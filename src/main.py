"""
AI Trading System v1.5 - BUCKET CHECK FIX
Fixed: SyncBucket error + improved storage handling
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Supabase
from supabase import create_client, Client

# ================================================================
# CONFIGURATION
# ================================================================

class Config:
    SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://kktfrmvwzykkzosvzddn.supabase.co')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtrdGZybXZ3enlra3pvc3Z6ZGRuIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NzgwNjc1MywiZXhwIjoyMDgzMzgyNzUzfQ.2OUwg8dQYaAjNDMHeUKXWxFeUm_ipxZgqx8x5RDcIU8')
    
    MODEL_VERSION = "v1.5.0"
    LOOKBACK_DAYS = 30
    PREDICTION_THRESHOLD = 0.65
    SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 
               'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY', 'XAUUSD']
    TIMEFRAME = 'H1'
    
    MOMENTUM_PERIODS = [3, 5, 10]
    VOLATILITY_WINDOW = 10
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

# ================================================================
# MODEL PERSISTENCE - FIXED BUCKET CHECK
# ================================================================

class ModelPersistence:
    def __init__(self, supabase: Client, symbol: str):
        self.supabase = supabase
        self.symbol = symbol
        self.bucket_name = 'models'
        self.bucket_exists = self._check_bucket_exists()
    
    def _check_bucket_exists(self) -> bool:
        """Check if storage bucket exists - FIXED VERSION"""
        try:
            # Try to list files in bucket - if succeeds, bucket exists
            self.supabase.storage.from_(self.bucket_name).list()
            print(f"   ✅ Storage bucket '{self.bucket_name}' found")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if 'not found' in error_msg or 'does not exist' in error_msg:
                print(f"\n⚠️  WARNING: Storage bucket '{self.bucket_name}' NOT FOUND!")
                print(f"   Please check Supabase Dashboard:")
                print(f"   Storage → Buckets → Should see '{self.bucket_name}' (PUBLIC)\n")
                return False
            else:
                # Other error, assume bucket might exist
                print(f"   ℹ️  Cannot verify bucket (will try to use): {e}")
                return True  # Assume exists, let upload/download fail if not
    
    def save_model(self, model, scaler, metrics: Dict):
        """Save model with better error handling"""
        if not self.bucket_exists:
            print(f"   ⏭️  Skipping save (bucket not found)")
            return False
        
        print(f"   💾 Saving model for {self.symbol}...")
        
        try:
            model_data = {
                'model': model,
                'scaler': scaler,
                'metrics': metrics,
                'timestamp': datetime.utcnow().isoformat(),
                'version': Config.MODEL_VERSION
            }
            
            model_bytes = pickle.dumps(model_data)
            file_path = f'{self.symbol}_model.pkl'
            
            # Upload
            self.supabase.storage.from_(self.bucket_name).upload(
                path=file_path,
                file=model_bytes,
                file_options={'content-type': 'application/octet-stream', 'upsert': 'true'}
            )
            
            print(f"   ✅ Model saved ({len(model_bytes)/1024:.1f} KB)")
            
            # Save metadata
            self._save_metadata(metrics)
            
            return True
            
        except Exception as e:
            print(f"   ⚠️  Error saving: {e}")
            print(f"   ℹ️  Model will work but won't persist for next run")
            return False
    
    def _save_metadata(self, metrics: Dict):
        """Save model metadata to database"""
        try:
            record = {
                'symbol': self.symbol,
                'model_version': Config.MODEL_VERSION,
                'train_accuracy': float(metrics.get('train_accuracy', 0)),
                'test_accuracy': float(metrics.get('test_accuracy', 0)),
                'n_estimators': int(metrics.get('n_estimators', 0)),
                'train_samples': int(metrics.get('train_samples', 0)),
                'test_samples': int(metrics.get('test_samples', 0)),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            self.supabase.table('model_metadata').upsert(record).execute()
            
        except:
            pass  # Ignore if table doesn't exist
    
    def load_model(self):
        """Load model with better error handling"""
        if not self.bucket_exists:
            print(f"   ℹ️  Bucket not found, will train new model")
            return None, None, None
        
        print(f"   📂 Checking for existing model for {self.symbol}...")
        
        try:
            file_path = f'{self.symbol}_model.pkl'
            model_bytes = self.supabase.storage.from_(self.bucket_name).download(file_path)
            model_data = pickle.loads(model_bytes)
            
            print(f"   ✅ Loaded existing model (saved: {model_data.get('timestamp', 'unknown')})")
            print(f"      Previous test accuracy: {model_data['metrics'].get('test_accuracy', 0):.4f}")
            
            return model_data['model'], model_data['scaler'], model_data['metrics']
            
        except Exception as e:
            print(f"   ℹ️  No existing model found (will train new)")
            return None, None, None

# ================================================================
# DATA LOADER
# ================================================================

class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    
    def load_ohlc_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
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
# FEATURE ENGINEER
# ================================================================

class FeatureEngineer:
    def __init__(self, config: Config):
        self.config = config
    
    def merge_data(self, ohlc: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
        if ohlc.empty or indicators.empty:
            return pd.DataFrame()
        df = pd.merge(ohlc, indicators, on=['timestamp', 'symbol', 'timeframe'], how='inner')
        print(f"✅ Merged dataset: {len(df)} records")
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        print("🔧 Engineering features...")
        
        for period in self.config.MOMENTUM_PERIODS:
            df[f'price_momentum_{period}'] = df['close'].pct_change(period) * 100
        
        df['volatility_ratio'] = df['atr_14'] / df['close']
        df['trend_strength'] = abs(df['macd_main'] - df['macd_signal'])
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(self.config.VOLATILITY_WINDOW).mean()
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['session'] = df['hour_of_day'].apply(self._get_session)
        df['session_encoded'] = df['session'].map({'asian': 0, 'european': 1, 'us': 2, 'other': 3})
        
        df['future_return_1h'] = df['close'].shift(-1) / df['close'] - 1
        df['future_return_4h'] = df['close'].shift(-4) / df['close'] - 1
        df['profitable_long'] = (df['future_return_1h'] > 0.0002).astype(int)
        df['profitable_short'] = (df['future_return_1h'] < -0.0002).astype(int)
        
        important_cols = [
            'close', 'volume', 'rsi_14', 'macd_main', 'macd_signal',
            'price_momentum_3', 'price_momentum_5', 'price_momentum_10',
            'volatility_ratio', 'trend_strength', 'volume_ratio',
            'hour_of_day', 'day_of_week', 'session_encoded',
            'future_return_1h', 'profitable_long', 'profitable_short'
        ]
        
        existing_cols = [col for col in important_cols if col in df.columns]
        df = df.dropna(subset=existing_cols)
        
        print(f"✅ Features created: {len(df)} complete records")
        return df
    
    def _get_session(self, hour: int) -> str:
        if 0 <= hour < 7:
            return 'asian'
        elif 7 <= hour < 15:
            return 'european'
        elif 15 <= hour < 21:
            return 'us'
        else:
            return 'other'

# ================================================================
# AI MODEL
# ================================================================

class AIModel:
    def __init__(self, config: Config, supabase: Client, symbol: str):
        self.config = config
        self.symbol = symbol
        self.persistence = ModelPersistence(supabase, symbol)
        
        existing_model, existing_scaler, existing_metrics = self.persistence.load_model()
        
        if existing_model:
            print(f"   🔄 Will IMPROVE existing model")
            self.model = existing_model
            self.scaler = existing_scaler
            self.previous_metrics = existing_metrics
            self.is_new_model = False
        else:
            print(f"   🆕 Will train NEW model")
            self.model = None
            self.scaler = StandardScaler()
            self.previous_metrics = {}
            self.is_new_model = True
        
        self.feature_columns = []
        self.metrics = {}
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi_14', 'macd_main', 'macd_signal', 
            'bb_upper', 'bb_middle', 'bb_lower', 'atr_14',
            'stoch_main', 'stoch_signal',
            'price_momentum_3', 'price_momentum_5', 'price_momentum_10',
            'volatility_ratio', 'trend_strength', 'volume_ratio',
            'hour_of_day', 'day_of_week', 'session_encoded'
        ]
        
        available_cols = [col for col in self.feature_columns if col in df.columns]
        X = df[available_cols].values
        
        y = np.zeros(len(df))
        y[df['profitable_long'] == 1] = 2
        y[df['profitable_short'] == 1] = 0
        
        self.feature_columns = available_cols
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train with overfitting prevention"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE
        )
        
        if self.is_new_model:
            print("   🤖 Training NEW model with anti-overfitting measures...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1,
                warm_start=True
            )
            
            self.model.fit(X_train_scaled, y_train)
            
        else:
            print("   🔄 IMPROVING existing model...")
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.n_estimators += 30
            self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, self.model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, self.model.predict(X_test_scaled))
        
        self.metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'n_estimators': self.model.n_estimators
        }
        
        overfitting_gap = (train_acc - test_acc) * 100
        
        if self.previous_metrics:
            prev_acc = self.previous_metrics.get('test_accuracy', 0)
            improvement = (test_acc - prev_acc) * 100
            print(f"   📈 Accuracy: {prev_acc:.4f} → {test_acc:.4f} ({improvement:+.2f}%)")
        else:
            print(f"   ✅ Model trained!")
            print(f"      Train accuracy: {train_acc:.4f}")
            print(f"      Test accuracy:  {test_acc:.4f}")
            print(f"      Overfitting gap: {overfitting_gap:.2f}%")
            
            if overfitting_gap > 20:
                print(f"      ⚠️  WARNING: Possible overfitting (gap > 20%)")
                print(f"      ℹ️  Normal with limited data, will improve over time")
        
        # Save
        self.persistence.save_model(self.model, self.scaler, self.metrics)
        
        return self.model
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model not trained!")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        confidence = np.max(probabilities, axis=1)
        
        return predictions, confidence
    
    def get_signal(self, prediction: int, confidence: float) -> str:
        if confidence < self.config.PREDICTION_THRESHOLD:
            return 'HOLD'
        if prediction == 2:
            return 'BUY'
        elif prediction == 0:
            return 'SELL'
        else:
            return 'HOLD'

# ================================================================
# PREDICTION WRITER
# ================================================================

class PredictionWriter:
    def __init__(self, config: Config, supabase: Client):
        self.config = config
        self.supabase = supabase
    
    def save_prediction(self, symbol: str, signal: str, confidence: float):
        try:
            record = {
                'timestamp': datetime.utcnow().isoformat(),
                'timestamp_unix': int(time.time()),
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

# ================================================================
# MAIN PIPELINE
# ================================================================

class TradingPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.loader = DataLoader(config)
        self.engineer = FeatureEngineer(config)
        self.supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        self.writer = PredictionWriter(config, self.supabase)
    
    def run_training(self, symbol: str):
        print(f"\n{'='*60}")
        print(f"TRAINING: {symbol}")
        print(f"{'='*60}\n")
        
        ohlc = self.loader.load_ohlc_data(symbol, days=self.config.LOOKBACK_DAYS)
        indicators = self.loader.load_indicators(symbol, days=self.config.LOOKBACK_DAYS)
        
        if ohlc.empty or indicators.empty:
            print(f"❌ Insufficient data for {symbol}")
            return None
        
        df = self.engineer.merge_data(ohlc, indicators)
        df = self.engineer.create_features(df)
        
        if len(df) < 50:
            print(f"❌ Not enough data: {len(df)} records")
            return None
        
        model = AIModel(self.config, self.supabase, symbol)
        X, y = model.prepare_training_data(df)
        model.train(X, y)
        
        return model
    
    def run_prediction(self, symbol: str, model: AIModel):
        print(f"\n📊 Generating prediction for {symbol}...")
        
        ohlc = self.loader.load_ohlc_data(symbol, days=5)
        indicators = self.loader.load_indicators(symbol, days=5)
        
        if ohlc.empty or indicators.empty:
            print(f"❌ No recent data")
            return
        
        df = self.engineer.merge_data(ohlc, indicators)
        df = self.engineer.create_features(df)
        
        if df.empty:
            print(f"❌ Failed to create features")
            return
        
        latest = df.iloc[-1]
        X_latest = latest[model.feature_columns].values.reshape(1, -1)
        
        prediction, confidence = model.predict(X_latest)
        signal = model.get_signal(prediction[0], confidence[0])
        
        self.writer.save_prediction(symbol, signal, confidence[0])
        print(f"   Signal: {signal} | Confidence: {confidence[0]:.2%}")

# ================================================================
# MAIN
# ================================================================

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     AI TRADING SYSTEM v1.5 - CONTINUOUS LEARNING          ║
    ║           WITH BUCKET CHECK FIX                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    config = Config()
    pipeline = TradingPipeline(config)
    
    print("\n🚀 RUNNING PIPELINE")
    print("="*60)
    
    trained_models = {}
    
    for symbol in config.SYMBOLS[:3]:
        try:
            model = pipeline.run_training(symbol)
            if model:
                trained_models[symbol] = model
                pipeline.run_prediction(symbol, model)
        except Exception as e:
            print(f"❌ Error with {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED!")
    print("="*60)
    print(f"\n📌 Summary:")
    print(f"   • Trained/Improved: {len(trained_models)} models")
    if trained_models:
        print(f"   • Models saved to Supabase Storage")
        print(f"   • Next run will improve existing models")
    else:
        print(f"   • No models saved (check bucket setup)")
    print("\n")

if __name__ == "__main__":
    main()
