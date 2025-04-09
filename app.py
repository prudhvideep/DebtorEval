import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from joblib import load
import json
import os

# Title and page config
st.set_page_config(page_title="Debtor Classification Engine", layout="wide")
st.title("Debtor Classification Engine")

# Check if required files exist and load them
required_files = [
    "xgboost_output_strategy_model.json",
    "label_encoder.joblib",
    "tone_encoder.joblib", 
    "personality_encoder.joblib",
    "social_status_encoder.joblib",
    "metadata.json",
    "feature_columns.json",
]

missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"Missing required files: {', '.join(missing_files)}")
    st.info("Please run the training script first to generate these files.")
    st.stop()

# Load model and encoders
try:
    # Load the model
    model = xgb.XGBClassifier()
    model.load_model("xgboost_output_strategy_model.json")
    
    # Load encoders
    le = load('label_encoder.joblib')  # Target encoder
    le_tone = load('tone_encoder.joblib')
    le_personality = load('personality_encoder.joblib')
    le_social_status = load('social_status_encoder.joblib')
    
    # Load metadata
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
        
    # Load feature columns order
    with open('feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
        
except Exception as e:
    st.error(f"Error loading model or encoders: {e}")
    st.stop()

# Strategy mapping with descriptions
strategy_map = {
    "Forgetful/Inattentive Payers": {
        "channel": ["WhatsApp", "SMS"],
        "tone": "Friendly Reminder",
        "frequency": "Weekly",
        "firmness": 3,
        "description": "These customers simply forget to pay but have good intention and payment capacity. Gentle reminders are usually sufficient."
    },
    "Financially Distressed Debtors": {
        "channel": ["IVR", "SMS"],
        "tone": "Empathetic",
        "frequency": "Bi-Weekly",
        "firmness": 2,
        "description": "These customers want to pay but are facing financial hardship. Empathetic communication and flexible payment plans are effective."
    },
    "Strategic Defaulters": {
        "channel": ["Legal Notice", "IVR"],
        "tone": "Formal",
        "frequency": "Daily",
        "firmness": 5,
        "description": "These customers have capacity to pay but choose not to. Formal communication and swift escalation to legal channels are required."
    },
    "Unreachable/Disengaged": {
        "channel": ["Field Agent Visit", "Registered Mail"],
        "tone": "Official",
        "frequency": "Monthly",
        "firmness": 4,
        "description": "These customers actively avoid communication. Multiple channels and official tone are needed for reconnection."
    },
    "Habitual Late Payers": {
        "channel": ["WhatsApp", "IVR"],
        "tone": "Assertive",
        "frequency": "Twice Weekly",
        "firmness": 4,
        "description": "These customers consistently pay late. Regular reminders with increasing firmness are effective."
    }
}

with st.form("debtor_form"):
    # Payment History Section
    st.header("📊 Payment History")
    col1, col2 = st.columns(2)
    with col1:
        on_time = st.number_input("On-time Payments (12m)", 0, 50, 10)
        late = st.number_input("Late Payments (12m)", 0, 20, 3)
        partial = st.number_input("Partial Payments (12m)", 0, 10, 1)
    with col2:
        defaults = st.number_input("Defaults Count", 0, 5, 0)
        age_of_account = st.number_input("Account Age (years)", 1, 10, 2)
        debt_amount = st.number_input("Debt Amount ($)", 0, 50000, 15000)

    # Demographics Section
    st.header("👤 Demographics")
    col3, col4 = st.columns(2)
    with col3:
        age = st.number_input("Age", 18, 100, 35)
        income = st.selectbox("Income Bracket", [15000, 30000, 50000, 70000])
    with col4:
        mpi = st.slider("MPI Index (1-100)", 1, 100, 50, 
                      help="Multi-Payment Index: Higher values indicate higher likelihood of strategic default")

    # Interaction Log Section
    st.header("📞 Interaction Log")
    col5, col6 = st.columns(2)
    with col5:
        whatsapp_rate = st.slider("WhatsApp Response Rate", 0.0, 1.0, 0.3)
        sms_rate = st.slider("SMS Response Rate", 0.0, 1.0, 0.2)
        ivr_rate = st.slider("IVR Response Rate", 0.0, 1.0, 0.4)
        email_rate = st.slider("Email Response Rate", 0.0, 1.0, 0.5)
    with col6:
        time_whatsapp = st.number_input("WhatsApp Response Time (hrs)", 1, 48, 12)
        time_sms = st.number_input("SMS Response Time (hrs)", 1, 48, 24)
        time_ivr = st.number_input("IVR Response Time (hrs)", 1, 48, 36)

    # Psychographic Signals Section
    st.header("🧠 Psychographic Signals")
    col7, col8 = st.columns(2)
    with col7:
        tone = st.selectbox("Communication Tone", 
                          options=metadata['tone_values'])
        personality = st.selectbox("Personality Type",
                                 options=metadata['personality_values'])
    with col8:
        social_status = st.selectbox("Social Status",
                                   options=metadata['social_status_values'])
        engagement = st.number_input("Engagement Score (10-80)", 10, 80, 45)

    submitted = st.form_submit_button("Classify Debtor")

if submitted:
    # Create progress bar
    progress_bar = st.progress(0)
    st.text("Processing debtor data...")
    
    # Encode categorical features
    tone_encoded = le_tone.transform([tone])[0]
    personality_encoded = le_personality.transform([personality])[0]
    social_status_encoded = le_social_status.transform([social_status])[0]
    
    progress_bar.progress(33)

    # Create input DataFrame with exact same columns as training data
    input_data = pd.DataFrame([{
        'on_time_payments': on_time,
        'late_payments': late,
        'partial_payments': partial,
        'defaults': defaults,
        'age_of_account': age_of_account,
        'debt_amount': debt_amount,
        'response_rate_whatsapp': whatsapp_rate,
        'response_rate_sms': sms_rate,
        'response_rate_ivr': ivr_rate,
        'time_to_respond_whatsapp': time_whatsapp,
        'time_to_respond_sms': time_sms,
        'time_to_respond_ivr': time_ivr,
        'age': age,
        'mpi_index': mpi,
        'income_bracket': income,
        'tone_analysis_encoded': tone_encoded,
        'personality_encoded': personality_encoded,
        'social_status_encoded': social_status_encoded,
        'engagement_score': engagement,
        'response_rate_email': email_rate,
        'Channel_encoded': 0,  
        'Tone_encoded': 0,     
        'Frequency_encoded': 0, 
        'Firmness_encoded': 0 
    }])
    
    progress_bar.progress(66)
    
    missing_columns = [col for col in feature_columns if col not in input_data.columns]
    if missing_columns:     
        st.warning(f"Missing columns in input data: {missing_columns}")
        # Add missing columns with default values
        for col in missing_columns:
            input_data[col] = 0  # Or any appropriate default value


    # Ensure column order matches training data
    input_data = input_data[feature_columns]
    
    # Make prediction
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)
    strategy = le.inverse_transform(prediction)[0]
    
    progress_bar.progress(100)
    
    # Create tabs for results
    tab1, tab2 = st.tabs(["Classification Results", "Debtor Profile Analysis"])
    
    with tab1:
        # Display results in a nicer format
        st.subheader("🔍 Classification Result")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"### {strategy}")
            st.markdown(f"*{strategy_map[strategy]['description']}*")
        with col2:
            st.metric("Confidence", f"{np.max(proba)*100:.1f}%")

        # Show communication strategy
        st.subheader("📨 Recommended Contact Strategy")
        
        col_channels, col_tone, col_freq, col_firm = st.columns(4)
        with col_channels:
            st.metric("Best Channels", ", ".join(strategy_map[strategy]['channel']))
        with col_tone:
            st.metric("Communication Tone", strategy_map[strategy]['tone'])
        with col_freq:
            st.metric("Contact Frequency", strategy_map[strategy]['frequency'])
        with col_firm:
            firmness_level = strategy_map[strategy]['firmness']
            st.metric("Firmness Level", f"{firmness_level}/5")
            
    with tab2:
        st.subheader("📊 Debtor Risk Profile")
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        # Calculate some derived metrics for display
        payment_ratio = on_time / (late + 1) # Adding 1 to avoid division by zero
        default_risk = (defaults * 20) + (late * 5)  # Simple formula to create a risk score
        response_avg = (whatsapp_rate + sms_rate + ivr_rate + email_rate) / 4
        
        with col1:
            st.metric("Payment History Score", f"{min(payment_ratio * 20, 100):.0f}/100")
            st.metric("Default Risk", f"{min(default_risk, 100):.0f}/100")
        
        with col2:
            st.metric("Communication Responsiveness", f"{response_avg*100:.0f}%")
            st.metric("Engagement Quality", f"{engagement:.0f}/80")
            
        with col3:
            st.metric("Debt to Income Ratio", f"{(debt_amount/income)*100:.1f}%")
            st.metric("Account Maturity", f"{age_of_account} years")
        
        # Show probabilities for all classes
        st.subheader("Alternative Classification Probabilities")
        
        # Create a dataframe for the probabilities
        proba_df = pd.DataFrame({
            'Category': [le.inverse_transform([i])[0] for i in range(len(proba[0]))],
            'Probability': proba[0]
        })
        proba_df = proba_df.sort_values('Probability', ascending=False)
        
        # Display bar chart of probabilities
        st.bar_chart(proba_df.set_index('Category'))