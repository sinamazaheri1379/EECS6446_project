# files/optimizations/scripts/train_capa_models.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import glob
import os

# 1. Find the latest baseline CSV
list_of_files = glob.glob('/home/common/EECS6446_project/files/optimizations/results/baseline_complete_*.csv')
latest_file = max(list_of_files, key=os.path.getctime)
print(f"Training on: {latest_file}")

data = pd.read_csv(latest_file)

# 2. Train Models for key services
# We map 'User Load' (Input) -> 'CPU Usage' (Output)
services = ['frontend', 'cartservice', 'checkoutservice']

os.makedirs('models', exist_ok=True)

for svc in services:
    # Features: Users, Time
    X = data[['scenario_users', 'elapsed_total_seconds']]
    # Target: CPU Millicores
    y = data[f'{svc}_cpu_millicores']

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # Save the "Brain"
    joblib.dump(model, f'models/{svc}_cpu_predictor.pkl')
    print(f"Saved model for {svc}")
