{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f6c8bf71-a954-4141-b3ce-697ef19ec68d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-04-21 21:38:08.985 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\ALEXIBJ\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-04-21 21:38:08.992 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# Load the trained model\n",
    "model = joblib.load(\"expresso_churn_model.pkl\")\n",
    "\n",
    "# Set page title\n",
    "st.set_page_config(page_title=\"Expresso Churn Prediction\", layout=\"centered\")\n",
    "\n",
    "st.title(\"📱 Expresso Churn Predictor\")\n",
    "st.markdown(\"Fill out the form below to predict if a customer is likely to churn.\")\n",
    "\n",
    "# Feature input form\n",
    "with st.form(\"prediction_form\"):\n",
    "    MONTANT = st.number_input(\"Recharge Amount (MONTANT)\", min_value=0.0, value=100.0)\n",
    "    REVENUE = st.number_input(\"Revenue\", min_value=0.0, value=120.0)\n",
    "    ARPU_SEGMENT = st.slider(\"ARPU Segment\", 1, 6, 3)\n",
    "    REGULARITY = st.slider(\"Regularity Score\", 0, 30, 15)\n",
    "    FREQUENCE_RECH = st.number_input(\"Recharge Frequency (FREQUENCE_RECH)\", min_value=0.0, value=5.0)\n",
    "    ON_NET = st.number_input(\"On-Net Minutes\", min_value=0.0, value=50.0)\n",
    "    DATA_VOLUME = st.number_input(\"Data Volume\", min_value=0.0, value=200.0)\n",
    "    ORANGE = st.number_input(\"Orange Minutes\", min_value=0.0, value=30.0)\n",
    "    FREQ_TOP_PACK = st.number_input(\"Top Pack Frequency\", min_value=0.0, value=2.0)\n",
    "\n",
    "    submit = st.form_submit_button(\"Predict\")\n",
    "\n",
    "# Collect inputs\n",
    "if submit:\n",
    "    features = np.array([[\n",
    "        REGION, MONTANT, REVENUE, ARPU_SEGMENT,\n",
    "        REGULARITY\n",
    "    ]])\n",
    "\n",
    "    prediction = model.predict(features)[0]\n",
    "    proba = model.predict_proba(features)[0][1]\n",
    "\n",
    "    st.markdown(\"### 🎯 Prediction Result:\")\n",
    "    if prediction == 1:\n",
    "        st.error(f\"🚨 This customer is likely to churn (Probability: {proba:.2f})\")\n",
    "    else:\n",
    "        st.success(f\"✅ This customer is likely to stay (Probability: {1 - proba:.2f})\")\n",
    "\n",
    "    st.markdown(\"🔎 Note: Probabilities closer to 1 indicate higher churn risk.\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
