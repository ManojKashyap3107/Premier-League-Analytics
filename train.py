from sklearn.linear_model import LinearRegression
import numpy as np

# Sample football data
# [Goals, Assists, Minutes]
X = np.array([
    [20, 10, 2500],
    [15, 8, 2200],
    [10, 12, 2000],
    [8, 5, 1800],
    [5, 4, 1500]
])

# Market values in millions of euros
y = np.array([120, 90, 70, 50, 35])

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# New player
new_player = [[18, 9, 2300]]

# Make prediction
prediction = model.predict(new_player)

print("===================================")
print(" PREMIER LEAGUE PLAYER PREDICTOR")
print("===================================")
print(f"Goals: 18")
print(f"Assists: 9")
print(f"Minutes: 2300")
print("-----------------------------------")
print(f"Predicted Market Value: €{prediction[0]:.2f} million")
print("===================================")