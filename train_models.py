import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading dataset...")

df = pd.read_csv("data/premier_league_players.csv")

print(f"Players loaded: {len(df)}")


# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

df["goals_per_90"] = (
    df["goals"] / df["minutes"] * 90
)

df["assists_per_90"] = (
    df["assists"] / df["minutes"] * 90
)

df["minutes_per_appearance"] = (
    df["minutes"] / df["appearances"]
)

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=[
        "age",
        "appearances",
        "minutes",
        "goals",
        "assists",
        "goals_per_90",
        "assists_per_90",
        "minutes_per_appearance",
        "position",
        "sub_position",
        "club",
        "market_value"
    ]
)


# ==========================================
# 3. FEATURES
# ==========================================

numeric_features = [
    "age",
    "appearances",
    "minutes",
    "goals",
    "assists",
    "goals_per_90",
    "assists_per_90",
    "minutes_per_appearance"
]

categorical_features = [
    "position",
    "sub_position",
    "club"
]

features = numeric_features + categorical_features

X = df[features]

# Log target
y = np.log1p(df["market_value"])


# ==========================================
# 4. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            categorical_features
        )
    ]
)


# ==========================================
# 5. TRAIN / TEST
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 6. MODELS
# ==========================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

    "Extra Trees":
        ExtraTreesRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            loss="huber",
            random_state=42
        )
}


# ==========================================
# 7. TRAIN MODELS
# ==========================================

results = []

best_model = None
best_pipeline = None
best_mae = float("inf")


for name, model in models.items():

    print()
    print("=" * 55)
    print(f"TRAINING: {name}")
    print("=" * 55)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    # Predict log value
    log_predictions = pipeline.predict(X_test)

    # Convert back to euros
    predictions = np.expm1(log_predictions)

    actual = np.expm1(y_test)

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    r2 = r2_score(
        actual,
        predictions
    )

    print(f"MAE : €{mae:,.0f}")
    print(f"RMSE: €{rmse:,.0f}")
    print(f"R²  : {r2:.3f}")

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    if mae < best_mae:

        best_mae = mae
        best_model = name
        best_pipeline = pipeline


# ==========================================
# 8. RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "MAE"
)

print()
print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))


# ==========================================
# 9. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_pipeline,
    "models/best_player_value_model.pkl"
)

print()
print(f"Best model: {best_model}")

print(
    "Saved: models/best_player_value_model.pkl"
)


# ==========================================
# 10. PREDICTIONS
# ==========================================

predictions = np.expm1(
    best_pipeline.predict(X_test)
)

prediction_df = df.loc[
    X_test.index,
    [
        "name",
        "club",
        "position",
        "market_value"
    ]
].copy()

prediction_df["predicted_value"] = predictions

prediction_df["difference"] = (
    prediction_df["predicted_value"]
    - prediction_df["market_value"]
)

prediction_df["absolute_error"] = (
    prediction_df["difference"].abs()
)

prediction_df = prediction_df.sort_values(
    "absolute_error"
)

prediction_df.to_csv(
    "data/best_model_predictions.csv",
    index=False
)

print()
print("Predictions saved:")
print("data/best_model_predictions.csv")