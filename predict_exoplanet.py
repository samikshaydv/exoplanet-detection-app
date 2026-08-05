import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ============================================================
# Load Trained Model
# ============================================================
model = load_model("models/hybrid_exoplanet_model.keras")

# ============================================================
# Load Datasets
# ============================================================
train = pd.read_csv("dataset/exoTrain_small.csv")
test = pd.read_csv("dataset/exoTest_small.csv")

# ============================================================
# Separate Features and Labels
# ============================================================
X_train = train.drop("LABEL", axis=1)
X_test = test.drop("LABEL", axis=1)

y_train = train["LABEL"]
y_test = test["LABEL"]

# ============================================================
# Convert Labels
# 1 -> 0 (No Exoplanet)
# 2 -> 1 (Exoplanet)
# ============================================================
y_train = y_train.replace({1: 0, 2: 1})
y_test = y_test.replace({1: 0, 2: 1})

# ============================================================
# Feature Scaling
# ============================================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# Reshape Data for CNN + BiLSTM
# ============================================================
X_train_scaled = X_train_scaled.reshape(
    X_train_scaled.shape[0],
    X_train_scaled.shape[1],
    1
)

X_test_scaled = X_test_scaled.reshape(
    X_test_scaled.shape[0],
    X_test_scaled.shape[1],
    1
)

# ============================================================
# Project Header
# ============================================================
print("\n" + "=" * 60)
print("         HYBRID CNN + BiLSTM EXOPLANET DETECTOR")
print("=" * 60)
print("Dataset : NASA Kepler Light Curve Dataset")
print("Model   : Hybrid CNN + BiLSTM")
print("Author  : Samiksha")
print("=" * 60)

# ============================================================
# User Input
# ============================================================
while True:
    try:
        import sys
        star_index = int(sys.stdin.readline().strip())


        if 0 <= star_index < len(X_test_scaled):
            break
        else:
            print("Please enter a number between 0 and 399.")

    except ValueError:
        print("Please enter a valid integer.")

# ============================================================
# Prediction
# ============================================================
sample = X_test_scaled[star_index].reshape(1, X_test_scaled.shape[1], 1)

probability = model.predict(sample, verbose=0)[0][0]

prediction = 1 if probability >= 0.5 else 0

confidence = probability if prediction == 1 else (1 - probability)

actual_label = y_test.iloc[star_index]
# ==============================
# Light Curve Graph
# ==============================

light_curve = X_test.iloc[star_index]

# ==========================================================
# Light Curve Visualization
# ==========================================================

light_curve = X_test.iloc[star_index].values

plt.figure(figsize=(14,5))

plt.plot(
    light_curve,
    color="royalblue",
    linewidth=1.5,
    label="Brightness"
)

plt.title(
    f"Light Curve Analysis | Star {star_index}\n"
    f"Prediction Confidence: {confidence*100:.2f}%",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Time", fontsize=12)
plt.ylabel("Relative Brightness", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.5)

plt.legend()

plt.tight_layout()

#plt.show()
# ============================================================
# Display Result
# ============================================================
print("\n")
print("=" * 60)
print("               EXOPLANET DETECTION RESULT")
print("=" * 60)

print(f"\nSelected Star Index : {star_index}")

if prediction == 1:
    print("Prediction          :  Exoplanet Detected")
else:
    print("Prediction          :  No Exoplanet")

print(f"Confidence Score    : {confidence * 100:.2f}%")
plt.figure(figsize=(5,1.5))
plt.barh(["Confidence"], [confidence*100])
plt.xlim(0,100)
plt.title("Model Confidence")
#plt.show()

if actual_label == 1:
    print("Actual Label        :  Exoplanet")
else:
    print("Actual Label        :  No Exoplanet")

if prediction == actual_label:
    print("\nStatus              :  CORRECT PREDICTION")
else:
    print("\nStatus              :  INCORRECT PREDICTION")

print("=" * 60)
print("\nAI Interpretation")

if prediction == 1:
    print("The light curve shows a periodic brightness dip.")
    print("The trained Hybrid CNN + BiLSTM model predicts that an Exoplanet is likely present.")
else:
    print("The light curve does not show a strong periodic transit pattern.")
    print("The trained Hybrid CNN + BiLSTM model predicts that no Exoplanet is present.")
