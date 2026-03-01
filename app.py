import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Credit Risk AI", page_icon="🏦", layout="wide")

# 2. Load Models Safely
@st.cache_resource
def load_models():
    lr = joblib.load('logistic_model.pkl')
    rf = joblib.load('random_forest_model.pkl')
    return lr, rf

try:
    lr_model, rf_model = load_models()
except Exception as e:
    st.error("⚠️ Model files not found. Please run train.py first.")
    st.stop()

# 3. Sidebar Navigation & Model Selection
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80) # Add a sleek icon
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["📊 Overview", "🔮 Prediction", "🧪 Testing"])

st.sidebar.divider()

st.sidebar.title("Model Settings")
model_choice = st.sidebar.selectbox(
    "Active Model Engine:", 
    ["Random Forest (Recommended)", "Logistic Regression (Baseline)"]
)

# --- PAGE 1: OVERVIEW ---
if page == "📊 Overview":
    st.title("🏦 Intelligent Credit Risk Scoring")
    st.markdown("### Mid-Sem Milestone 1: Predictive Analytics System")
    st.write("Welcome to the automated lending decision support system. This platform utilizes machine learning to evaluate borrower profiles and predict default probabilities.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Size", "32,581 Records")
    col2.metric("RF Model Accuracy", "93.31%")
    col3.metric("LR Model Accuracy", "81.51%")
    
    st.divider()
    st.markdown("""
    #### ⚙️ System Architecture
    * **Preprocessing Pipeline:** Outlier clipping (IQR), Median Imputation, One-Hot Encoding.
    * **Model 1:** Scikit-Learn Random Forest Classifier (Class-weighted).
    * **Model 2:** Scikit-Learn Logistic Regression.
    * **Evaluation Metric:** ROC-AUC optimized to handle class imbalances.
    """)

# --- PAGE 2: PREDICTION ---
elif page == "🔮 Prediction":
    st.title("🔮 Single Borrower Assessment")
    st.write("Enter the borrower's details below to generate an instant risk assessment.")
    
    with st.form("prediction_form"):
        st.subheader("Borrower Financial Profile")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", 18, 100, 30)
            income = st.number_input("Annual Income ($)", 0, 1000000, 50000, step=1000)
            emp_len = st.number_input("Employment Length (Years)", 0, 50, 5)
            
        with col2:
            home = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
            intent = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
            grade = st.selectbox("Assigned Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
            
        with col3:
            loan_amt = st.number_input("Requested Loan Amount ($)", 0, 100000, 10000, step=500)
            rate = st.number_input("Interest Rate (%)", 0.0, 30.0, 10.5)
            hist = st.number_input("Credit History (Years)", 0, 50, 5)
            default = st.selectbox("Historical Default?", ["N", "Y"])
            
        submit = st.form_submit_button("Run Risk Analysis", use_container_width=True)

    if submit:
        loan_percent_income = loan_amt / income if income > 0 else 0
        input_df = pd.DataFrame([[
            age, income, emp_len, loan_amt, rate, loan_percent_income, hist, home, intent, grade, default
        ]], columns=['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 
                     'loan_percent_income', 'cb_person_cred_hist_length', 'person_home_ownership', 
                     'loan_intent', 'loan_grade', 'cb_person_default_on_file'])

        active_model = lr_model if "Logistic" in model_choice else rf_model
        prediction = active_model.predict(input_df)[0]
        prob = active_model.predict_proba(input_df)[0][1]

        st.divider()
        st.subheader("Assessment Results")
        
        if prediction == 1:
            st.error("🚨 **DECISION: REJECT**")
            st.write(f"The model has flagged this profile as **HIGH RISK**. There is a **{prob:.2%}** probability of default.")
            st.progress(float(prob))
        else:
            st.success("✅ **DECISION: APPROVE**")
            st.write(f"The model has flagged this profile as **LOW RISK**. The probability of default is only **{prob:.2%}**.")
            st.progress(float(prob))

# --- PAGE 3: TESTING ---
# --- PAGE 3: TESTING ---
elif page == "🧪 Testing":
    st.title("🧪 Bulk Model Testing & Validation")
    st.write("Upload a CSV file containing multiple borrower profiles to predict their risk simultaneously.")
    
    active_model = lr_model if "Logistic" in model_choice else rf_model
    st.info(f"Currently scoring using: **{model_choice}**")
    
    # File uploader widget
    uploaded_file = st.file_uploader("Upload Borrower Data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # 1. Read the uploaded CSV
            test_df = pd.read_csv(uploaded_file)
            
            # 2. Check if the target column is in the file and drop it if necessary
            if 'loan_status' in test_df.columns:
                features_df = test_df.drop('loan_status', axis=1)
            else:
                features_df = test_df.copy()
                
            # 3. Ensure the columns match exactly what the model expects
            expected_cols = [
                'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 
                'loan_percent_income', 'cb_person_cred_hist_length', 'person_home_ownership', 
                'loan_intent', 'loan_grade', 'cb_person_default_on_file'
            ]
            
            # Extract only the needed columns
            input_df = features_df[expected_cols]
            
            # 4. Make Predictions in bulk
            with st.spinner("Analyzing borrower profiles..."):
                predictions = active_model.predict(input_df)
                probabilities = active_model.predict_proba(input_df)[:, 1]
            
            # 5. Format the results
            results_df = test_df.copy()
            results_df['Default_Probability'] = probabilities
            # Create a clean visually appealing Decision column
            results_df['Decision'] = ["❌ REJECTED (High Risk)" if p == 1 else "✅ APPROVED (Low Risk)" for p in predictions]
            
            st.success(f"Successfully scored {len(results_df)} borrowers!")
            
            # 6. Display the scrollable table
            st.write("### Batch Assessment Results")
            # We reorganize the columns so the Decision is the very first thing they see
            display_cols = ['Decision', 'Default_Probability'] + expected_cols
            st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)
            
            # 7. Add a download button for the results
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Scored Results (CSV)",
                data=csv_data,
                file_name='batch_scored_borrowers.csv',
                mime='text/csv',
            )
            
        except KeyError as e:
            st.error(f"⚠️ Error: The uploaded CSV is missing a required column: {e}")
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred: {e}")