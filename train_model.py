import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# ==========================================
# 1. LOAD DATA
# ==========================================

DATA_FILE = "data/premier_league_players.csv"

print("Loading Premier League dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Loaded {len(df)} players.")


# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

# Avoid division by zero
df["goals_per_90"] = (
    df["goals"] / df["minutes"] * 90
)

df["assists_per_90"] = (
    df["assists"] / df["minutes"] * 90
)

# Replace possible infinity values
df = df.replace([np.inf, -np.inf], np.nan)

# Remove rows with missing values
df = df.dropna(
    subset=[
        "age",
        "appearances",
        "minutes",
        "goals",
        "assists",
        "goals_per_90",
        "assists_per_90",
        "market_value"
    ]
)


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

features = [
    "age",
    "appearances",
    "minutes",
    "goals",
    "assists",
    "goals_per_90",
    "assists_per_90"
]

X = df[features]

y = df["market_value"]


print()
print("Features:")
print(features)

print()
print(f"Training samples: {len(df)}")


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print()
print(f"Training players: {len(X_train)}")
print(f"Testing players: {len(X_test)}")


# ==========================================
# 5. CREATE MODEL
# ==========================================

print()
print("Training Linear Regression model...")

model = LinearRegression()

model.fit(X_train, y_train)


# ==========================================
# 6. PREDICTIONS
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 7. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print()
print("==========================================")
print(" MODEL RESULTS")
print("==========================================")

print(f"MAE : €{mae:,.0f}")
print(f"RMSE: €{rmse:,.0f}")
print(f"R²  : {r2:.3f}")


# ==========================================
# 8. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/player_value_model.pkl"
)

print()
print("Model saved:")
print("models/player_value_model.pkl")


# ==========================================
# 9. SAVE PREDICTIONS
# ==========================================

results = df.loc[
    X_test.index,
    [
        "name",
        "club",
        "position",
        "market_value"
    ]
].copy()

results["predicted_value"] = predictions

results["difference"] = (
    results["predicted_value"]
    - results["market_value"]
)

results = results.sort_values(
    "market_value",
    ascending=False
)

results.to_csv(
    "data/model_predictions.csv",
    index=False
)

print()
print("Predictions saved:")
print("data/model_predictions.csv")


# ==========================================
# 10. SHOW EXAMPLES
# ==========================================

print()
print("==========================================")
print(" SAMPLE PREDICTIONS")
print("==========================================")

for _, row in results.head(10).iterrows():

    actual = row["market_value"]
    predicted = row["predicted_value"]

    print(
        f"{row['name']:<25} "
        f"Actual: €{actual:>12,.0f}   "
        f"Predicted: €{predicted:>12,.0f}"
    )