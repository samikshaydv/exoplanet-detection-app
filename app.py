import streamlit as st
import subprocess
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# -------------------------
# Load Dataset
# -------------------------
test = pd.read_csv("dataset/exoTest_small.csv")
X_test = test.drop("LABEL", axis=1)

# -------------------------
# Streamlit Page Settings
# -------------------------
st.set_page_config(
    page_title="Exoplanet Detection",
    page_icon="🪐",
    layout="wide"
)

# -------------------------
# Title
# -------------------------
st.title("🪐 Exoplanet Detection System")
st.subheader("Hybrid CNN + BiLSTM using NASA Kepler Light Curve Data")

st.markdown("""
This application uses a **Hybrid CNN + BiLSTM Deep Learning Model**
trained on the **NASA Kepler Light Curve Dataset**
to predict whether a selected star contains an **Exoplanet**.
""")

st.write("---")

# -------------------------
# Model Information
# -------------------------
st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:
    st.info("""
**Dataset**

NASA Kepler Light Curve Dataset

**Architecture**

Hybrid CNN + BiLSTM
""")

with col2:
    st.info("""
**Author**

Samiksha

**Task**

Exoplanet Detection
""")

st.write("---")

# -------------------------
# Select Star
# -------------------------
star = st.slider(
    "Select Star Index",
    min_value=0,
    max_value=399,
    value=0
)
# -------------------------
# Prediction Button
# -------------------------

if st.button("🔍 Predict"):

    with st.spinner("Running Hybrid CNN + BiLSTM Model..."):

        result = subprocess.run(
            ["python", "predict_exoplanet.py"],
            input=str(star),
            text=True,
            capture_output=True
        )

    output = result.stdout
    error = result.stderr

    if result.returncode != 0:
        st.error("Prediction script failed.")
        st.code(error)
        st.stop()

    # -------------------------
    # Prediction
    # -------------------------

    prediction = "No Exoplanet"

    if "Exoplanet Detected" in output:
        prediction = "Exoplanet Detected"

    if prediction == "Exoplanet Detected":
        st.success("🪐 Prediction: Exoplanet Detected")
    else:
        st.error("⭐ Prediction: No Exoplanet")

    # -------------------------
    # Confidence Score Extraction
    # -------------------------

    confidence = None

    try:
        for line in output.splitlines():

            if "Confidence Score" in line:

                confidence = float(
                    line.split(":")[1]
                    .replace("%", "")
                    .strip()
                )

                break

    except Exception:
        confidence = None
            # -------------------------
    # Light Curve Visualization
    # -------------------------

    st.write("---")
    st.subheader("📈 Kepler Light Curve")

    # Get the real light curve of the selected star
    light_curve = X_test.iloc[star].values

    # X-axis (time points)
    time = np.arange(len(light_curve))

    # Y-axis (brightness values)
    flux = light_curve

    # Create Plotly Figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time,
            y=flux,
            mode="lines",
            name="Brightness",
            line=dict(
                color="royalblue",
                width=2
            )
        )
    )

    fig.update_layout(
        title=f"Light Curve of Star {star}",
        xaxis_title="Time Index",
        yaxis_title="Normalized Brightness",
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)
        # -------------------------
    # Confidence Score
    # -------------------------

    st.write("---")
    st.subheader("📊 Confidence Score")

    if confidence is not None:

        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(
            min(max(confidence / 100, 0), 1)
        )

    else:
        st.warning("Confidence score not available.")

    # -------------------------
    # Prediction Details
    # -------------------------

    st.write("---")
    st.subheader("📋 Prediction Details")

    st.write(f"**Selected Star Index:** {star}")

    actual = "Unknown"
    status = "Unknown"

    for line in output.splitlines():

        if "Actual Label" in line:
            actual = line.split(":")[1].strip()

        elif "Status" in line:
            status = line.split(":")[1].strip()

    st.write(f"**Actual Label:** {actual}")

    if "CORRECT" in status.upper():
        st.success(status)
    else:
        st.warning(status)

    # -------------------------
    # AI Interpretation
    # -------------------------

    st.write("---")
    st.subheader("🤖 AI Interpretation")

    interpretation = ""

    if "AI Interpretation" in output:

        interpretation = output.split(
            "AI Interpretation", 1
        )[1].strip()

    if interpretation:
        st.info(interpretation)
    else:
        st.info(
            "The model analyzed the selected star using the "
            "Hybrid CNN + BiLSTM model."
        )

    # -------------------------
    # Raw Model Output
    # -------------------------

    with st.expander("📄 View Raw Model Output"):
        st.code(output)
