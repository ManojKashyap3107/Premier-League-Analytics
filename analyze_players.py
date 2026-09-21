import pandas as pd

# Load model predictions
df = pd.read_csv("data/best_model_predictions.csv")

# Difference between model and listed market value
df["difference_m"] = (
    df["difference"] / 1_000_000
)

df["actual_m"] = (
    df["market_value"] / 1_000_000
)

df["predicted_m"] = (
    df["predicted_value"] / 1_000_000
)

# Percentage difference
df["difference_percent"] = (
    df["difference"]
    / df["market_value"]
    * 100
)

print()
print("=" * 75)
print(" PREMIER LEAGUE PLAYER VALUE ANALYSIS")
print("=" * 75)


# ==========================================
# MODEL ABOVE LISTED VALUE
# ==========================================

print()
print("MODEL-IMPLIED VALUE ABOVE LISTED VALUE")
print("-" * 75)

above = df.sort_values(
    "difference",
    ascending=False
).head(15)

for _, row in above.iterrows():

    print(
        f"{row['name']:<25}"
        f" Listed: €{row['actual_m']:>6.1f}M"
        f"  Model: €{row['predicted_m']:>6.1f}M"
        f"  Difference: +€{row['difference_m']:>6.1f}M"
    )


# ==========================================
# MODEL BELOW LISTED VALUE
# ==========================================

print()
print("MODEL-IMPLIED VALUE BELOW LISTED VALUE")
print("-" * 75)

below = df.sort_values(
    "difference",
    ascending=True
).head(15)

for _, row in below.iterrows():

    print(
        f"{row['name']:<25}"
        f" Listed: €{row['actual_m']:>6.1f}M"
        f"  Model: €{row['predicted_m']:>6.1f}M"
        f"  Difference: €{row['difference_m']:>6.1f}M"
    )


# ==========================================
# SAVE ANALYSIS
# ==========================================

df.to_csv(
    "data/player_value_analysis.csv",
    index=False
)

print()
print("Analysis saved:")
print("data/player_value_analysis.csv")