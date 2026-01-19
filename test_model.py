import pandas as pd
import joblib

# =============================
# Load models & scalers
# =============================
fault_status_model = joblib.load("fault_status_model.pkl")
fault_type_model = joblib.load("fault_type_model.pkl")

scaler_status = joblib.load("scaler_status.pkl")
scaler_type = joblib.load("scaler_type.pkl")

# =============================
# Fault labels
# =============================
fault_map = {
    1: "Congestion",
    3: "Hardware Fault",
    4: "High Latency",
    5: "Packet Loss"
}

# =============================
# STRONG PACKET LOSS INPUT
# =============================
sample = pd.DataFrame([{
    'latency': 200,
    'packet_drops': 1100,
    'error_rate': 0.02,
    'retransmissions': 120,
    'utilization': 65,
    'in_traffic': 900,
    'out_traffic': 880
}])

print("\nTEST INPUT:")
print(sample)

# =============================
# MODEL-1 CHECK
# =============================
sample_scaled = scaler_status.transform(sample)
status = fault_status_model.predict(sample_scaled)[0]

if status == 0:
    print("\nRESULT: No Fault (Model-1 blocked)")
else:
    print("\nRESULT: Fault Detected (Model-1 passed)")

    sample_scaled_type = scaler_type.transform(sample)
    fault = fault_type_model.predict(sample_scaled_type)[0]
    print("FAULT TYPE →", fault_map[fault])

# =============================
# DIRECT MODEL-2 VERIFICATION
# =============================
print("\nDIRECT MODEL-2 CHECK:")
sample_scaled_type = scaler_type.transform(sample)
fault_direct = fault_type_model.predict(sample_scaled_type)[0]
print("FAULT TYPE →", fault_map[fault_direct])
