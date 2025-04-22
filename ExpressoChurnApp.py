import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("expresso_churn_model.pkl")

st.set_page_config(page_title="Expresso Churn Predictor", layout="centered")

st.title("📱 Expresso Churn Prediction App")
st.write("Enter customer data to predict the likelihood of churn.")

# Feature input form
with st.form("prediction_form"):
    MONTANT = st.number_input("Recharge Amount (MONTANT)", min_value=0.0, value=100.0)
    FREQUENCE_RECH = st.number_input("Recharge Frequency (FREQUENCE_RECH)", min_value=0.0, value=5.0)
    REVENUE = st.number_input("Revenue", min_value=0.0, value=120.0)
    ARPU_SEGMENT = st.slider("ARPU Segment", 1, 6, 3)
    DATA_VOLUME = st.number_input("Data Volume", min_value=0.0, value=200.0)
    ON_NET = st.number_input("On-Net Minutes", min_value=0.0, value=50.0)
    ORANGE = st.number_input("Orange Minutes", min_value=0.0, value=30.0)
    REGULARITY = st.slider("Regularity Score", 0, 30, 15)
    TOP_PACK = st.selectbox("Top Pack (encoded)", options=[0, 1])  # Must match model encoding

    submit = st.form_submit_button("Predict")

if submit:
    features = np.array([[
        REGULARITY, REVENUE, ARPU_SEGMENT,
        MONTANT, DATA_VOLUME, 12,  # TENURE is fixed or inferred
        FREQUENCE_RECH, 8,         # FREQUENCE fixed/default
        5                          # REGION default or one-hot encoded
    ]])

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    st.markdown("### 🧾 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Customer is likely to stay (Probability: {1 - proba:.2f})")

    st.caption("Note: Probabilities closer to 1 mean higher risk of churn.")