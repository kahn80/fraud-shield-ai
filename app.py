import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("="*60)
print("🛡️ FRAUD SHIELD AI - MODEL TRAINING")
print("="*60)

# Load data
print("\n📂 Loading dataset...")
df = pd.read_csv('creditcard.csv')
print(f"✅ Loaded {len(df):,} transactions")

# Prepare data
print("\n🔧 Preparing data...")
X = df.drop('Class', axis=1)  # ALL 30 features
y = df['Class']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Training: {len(X_train):,} | Testing: {len(X_test):,}")

# Scale ALL features (30 features)
print("\n📊 Scaling ALL 30 features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Trains on ALL 30 features
X_test_scaled = scaler.transform(X_test)

# Train
print("\n🤖 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train_scaled, y_train)
print("✅ Training complete!")

# Evaluate
print("\n📈 Evaluating model...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Save
print("\n💾 Saving model and scaler...")
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Saves scaler trained on ALL 30 features
print("✅ Model saved as 'fraud_model.pkl'")
print("✅ Scaler saved as 'scaler.pkl'")

print("\n" + "="*60)
print("✅ MODEL READY! Run: streamlit run ui.py")
print("="*60)