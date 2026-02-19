import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load trained rust model
model = tf.keras.models.load_model("keras_model.h5", compile=False)

# Load labels
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

st.title("🚲 Cycle Resale Price Estimator (Rust MVP)")

st.markdown("Upload cycle image + enter usage details")

# --------------------------
# USER INPUTS
# --------------------------

mrp = st.number_input("Enter MRP (₹)", min_value=0.0, step=500.0)
years_used = st.number_input("Years Used", min_value=0, step=1)
months_used = st.number_input("Additional Months Used", min_value=0, max_value=11, step=1)

uploaded_file = st.file_uploader("Upload Cycle Image", type=["jpg", "jpeg", "png"])

# --------------------------
# IMAGE PREDICTION
# --------------------------

rust_confidence = 0
predicted_label = "Not Evaluated"

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize image to 224x224 (Teachable Machine default)
    image = image.resize((224, 224))
    image_array = np.asarray(image)

    # Normalize
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    predicted_label = class_names[index]
    confidence_score = float(prediction[0][index])

    st.subheader("🧠 Rust Detection Result")
    st.write(f"Prediction: **{predicted_label}**")
    st.write(f"Confidence: **{round(confidence_score * 100, 2)}%**")

    # If predicted rust, take its confidence
    if "rust" in predicted_label.lower():
        rust_confidence = confidence_score
    else:
        rust_confidence = 0

# --------------------------
# PRICE CALCULATION
# --------------------------

if st.button("Calculate Final Price"):

    # Total usage in months
    total_months = years_used * 12 + months_used

    # Depreciation rate
    annual_rate = 0.15
    monthly_rate = annual_rate / 12

    depreciation_factor = (1 - monthly_rate) ** total_months
    depreciated_value = mrp * depreciation_factor

    # Rust deduction (max ₹500 scaled by confidence)
    rust_deduction = 500 * rust_confidence

    final_price = depreciated_value - rust_deduction

    if final_price < 0:
        final_price = 0

    st.subheader("💰 Price Breakdown")

    st.write(f"Depreciated Value: ₹ {round(depreciated_value, 2)}")
    st.write(f"Rust Deduction (Confidence Weighted): ₹ {round(rust_deduction, 2)}")
    st.write(f"### ✅ Final Estimated Price: ₹ {round(final_price, 2)}")

