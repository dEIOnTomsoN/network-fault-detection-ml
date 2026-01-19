import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# =============================
# 1. Load dataset
# =============================
df = pd.read_csv("network_dd.csv")

# =============================
# 2. RULE-BASED FAULT CREATION
# (Packet Loss has highest priority)
# =============================
def generate_fault(row):
    if row['packet_drops'] > 100 and row['error_rate'] < 0.05:
        return 5  # Packet Loss
    elif row['utilization'] > 85 and row['packet_drops'] > 50:
        return 1  # Congestion
    elif row['error_rate'] > 0.05:
        return 3  # Hardware Fault
    elif row['latency'] > 300:
        return 4  # High Latency
    else:
        return 0  # No Fault

df['fault_type'] = df.apply(generate_fault, axis=1)

# =============================
# 3. Fault Status
# =============================
df['fault_status'] = df['fault_type'].apply(lambda x: 0 if x == 0 else 1)

print("\nFault distribution:")
print(df['fault_type'].value_counts())

# =============================
# 4. Features
# =============================
features = [
    'latency',
    'packet_drops',
    'error_rate',
    'retransmissions',
    'utilization',
    'in_traffic',
    'out_traffic'
]

X = df[features]

# ==================================================
# MODEL-1 : FAULT / NO FAULT  (CLASS WEIGHT FIX)
# ==================================================
scaler_status = StandardScaler()
X_scaled = scaler_status.fit_transform(X)

y_status = df['fault_status']

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_status, test_size=0.3, random_state=42
)

fault_status_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight={0: 1, 1: 3}   # 🔴 IMPORTANT FIX
)

fault_status_model.fit(X_train, y_train)

print("✔ Model-1 (Fault Status) Trained")

joblib.dump(fault_status_model, "fault_status_model.pkl")
joblib.dump(scaler_status, "scaler_status.pkl")

# ==================================================
# MODEL-2 : FAULT TYPE
# ==================================================
fault_df = df[df['fault_status'] == 1]

X_fault = fault_df[features]
y_type = fault_df['fault_type']

scaler_type = StandardScaler()
X_fault_scaled = scaler_type.fit_transform(X_fault)

Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_fault_scaled, y_type, test_size=0.3, random_state=42
)

fault_type_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

fault_type_model.fit(Xf_train, yf_train)

print("✔ Model-2 (Fault Type) Trained")

joblib.dump(fault_type_model, "fault_type_model.pkl")
joblib.dump(scaler_type, "scaler_type.pkl")

print("✔ ALL MODELS SAVED SUCCESSFULLY")
