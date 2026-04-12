import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="EMI Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #4F46E5;
        --secondary-color: #10B981;
        --accent-color: #F59E0B;
        --danger-color: #EF4444;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.15);
    }
    
    .feature-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .feature-box h3 {
        color: #4F46E5;
        margin-bottom: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.3rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
        color: #10B981;
    }
    
    /* Success/Warning/Error boxes */
    .stSuccess, .stWarning, .stError {
        padding: 1rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    /* Input field styling */
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select,
    .stSlider>div>div>div>div {
        border-radius: 8px;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    .hero-section h1 {
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    
    .hero-section p {
        font-size: 1.4rem;
        opacity: 0.95;
        line-height: 1.6;
    }
    
    /* Stats box */
    .stats-box {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        margin: 1rem;
    }
    
    .stats-box h2 {
        color: #667eea;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 800;
    }
    
    .stats-box p {
        color: #666;
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='color: white; font-size: 2.5rem; margin: 0;'>💰</h1>
    <h2 style='color: white; margin: 0.5rem 0;'>EMI Predictor</h2>
    <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Intelligent Loan Assessment</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Check Eligibility"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='color: white; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; margin-top: 2rem;'>
    <h4 style='margin: 0 0 0.5rem 0;'>ℹ️ Quick Info</h4>
    <p style='font-size: 0.85rem; margin: 0; opacity: 0.9;'>
    This AI-powered system uses advanced machine learning to predict EMI eligibility with high accuracy.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# HOME PAGE
# =========================
if page == "🏠 Home":
    # Hero Section
    st.markdown("""
    <div class='hero-section'>
        <h1>🎯 EMI Eligibility Prediction System</h1>
        <p>Leverage cutting-edge AI to make smarter lending decisions</p>
        <p style='font-size: 1.1rem; margin-top: 1rem; opacity: 0.85;'>
            Powered by XGBoost Machine Learning Models
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='stats-box'>
            <h2>🎯</h2>
            <h2>95%+</h2>
            <p>Prediction Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='stats-box'>
            <h2>⚡</h2>
            <h2>&lt;2s</h2>
            <p>Processing Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='stats-box'>
            <h2>🔒</h2>
            <h2>100%</h2>
            <p>Data Security</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # About Section
    st.markdown("<div class='section-header'>📖 About This System</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <h3>🤖 AI-Powered Decision Making</h3>
        <p style='color: #555; line-height: 1.8; font-size: 1.05rem;'>
            Our EMI Eligibility Prediction System uses state-of-the-art machine learning algorithms 
            to analyze multiple financial and personal parameters. The system evaluates creditworthiness 
            by considering salary, expenses, credit score, employment history, and various other factors 
            to provide accurate EMI predictions and eligibility assessments.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    st.markdown("<div class='section-header'>✨ Key Features</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-box'>
            <h3>🎯 Multi-Factor Analysis</h3>
            <p>Comprehensive evaluation using 20+ parameters including income, expenses, credit history, and employment details</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='feature-box'>
            <h3>📊 Accurate Predictions</h3>
            <p>Advanced XGBoost models trained on extensive datasets for highly reliable eligibility classification</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='feature-box'>
            <h3>⚡ Instant Results</h3>
            <p>Get immediate feedback on loan eligibility and expected EMI amount within seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-box'>
            <h3>🔐 Risk Assessment</h3>
            <p>Three-tier classification: Eligible, High Risk, and Not Eligible for better decision making</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='feature-box'>
            <h3>💡 Smart Calculations</h3>
            <p>Intelligent DTI ratio, expense analysis, and credit utilization calculations for precision</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='feature-box'>
            <h3>🎨 User-Friendly</h3>
            <p>Intuitive interface with clear input fields and easy-to-understand results</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How It Works
    st.markdown("<div class='section-header'>⚙️ How It Works</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: white; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);'>
            <h2 style='color: #667eea; margin: 0;'>1️⃣</h2>
            <h4>Input Data</h4>
            <p style='color: #666; font-size: 0.9rem;'>Enter your financial and personal details</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: white; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);'>
            <h2 style='color: #667eea; margin: 0;'>2️⃣</h2>
            <h4>AI Analysis</h4>
            <p style='color: #666; font-size: 0.9rem;'>Machine learning models process your data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: white; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);'>
            <h2 style='color: #667eea; margin: 0;'>3️⃣</h2>
            <h4>Prediction</h4>
            <p style='color: #666; font-size: 0.9rem;'>Get eligibility status and EMI amount</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: white; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);'>
            <h2 style='color: #667eea; margin: 0;'>4️⃣</h2>
            <h4>Decision</h4>
            <p style='color: #666; font-size: 0.9rem;'>Make informed lending decisions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # CTA
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
        <h2 style='margin: 0 0 1rem 0;'>Ready to Check Eligibility? 🚀</h2>
        <p style='font-size: 1.1rem; margin: 0;'>Navigate to the "Check Eligibility" page using the sidebar</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# ELIGIBILITY CHECK PAGE
# =========================
elif page == "📊 Check Eligibility":
    
    # Load models
    try:
        clf_model = joblib.load("processed_data/xgb_classifier_final.pkl")
        reg_model = joblib.load("processed_data/xgb_regressor_final.pkl")
        clf_feature_cols = joblib.load("processed_data/feature_columns.pkl")
        reg_feature_cols = joblib.load("processed_data/reg_feature_columns.pkl")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()
    
    # Feature engineering function
    def engineer_features(df):
        df_new = df.copy()
        df_new = df_new.drop(columns=['dependents', 'emergency_fund'], errors='ignore')
        
        expense_cols = [
            'monthly_rent','school_fees','college_fees',
            'travel_expenses','groceries_utilities','other_monthly_expenses'
        ]
        df_new['total_expenses'] = df_new[expense_cols].sum(axis=1)
        df_new['dti_ratio'] = df_new['current_emi_amount'] / df_new['monthly_salary']
        
        skewed_cols = ['monthly_salary','requested_amount','bank_balance']
        for col in skewed_cols:
            df_new[f'{col}_log'] = np.log1p(df_new[col])
        
        df_new['loan_to_income'] = df_new['requested_amount_log'] / df_new['monthly_salary_log']
        df_new['available_for_emi'] = (
            df_new['monthly_salary'] - df_new['total_expenses'] - df_new['current_emi_amount']
        )
        df_new['expense_ratio'] = df_new['total_expenses'] / df_new['monthly_salary']
        df_new['emi_capacity'] = df_new['available_for_emi'] * 0.50
        df_new['credit_multiplier'] = (df_new['credit_score'] - 300) / 600
        df_new['adjusted_capacity'] = (
            df_new['emi_capacity'] * (0.8 + 0.4 * df_new['credit_multiplier'])
        )
        df_new['credit_utilization'] = (
            df_new['current_emi_amount'] / (df_new['monthly_salary'] + 1)
        ).clip(upper=1.0)
        df_new['credit_dti_interaction'] = df_new['credit_score'] * df_new['dti_ratio']
        df_new = df_new.drop(columns=skewed_cols)
        
        return df_new
    
    # Page header
    st.markdown("""
    <div class='main-header'>
        <h1>📊 EMI Eligibility Checker</h1>
        <p>Fill in your details below to check loan eligibility and estimated EMI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='section-header'>👤 Personal Information</div>", unsafe_allow_html=True)
        
        age = st.slider("Age", 18, 60, 35, help="Your current age")
        years_of_employment = st.slider("Years of Employment", 0, 30, 7, help="Total years of work experience")
        
        gender = st.selectbox("Gender", ["male","female"], index=0)
        marital_status = st.selectbox("Marital Status", ["single","married"], index=1)
        
        education = st.selectbox(
            "Education Level",
            ["high_school","graduate","post_graduate","professional"],
            index=1
        )
        
        employment_type = st.selectbox(
            "Employment Type",
            ["private","government","self_employed"],
            index=0
        )
        
        company_type = st.selectbox(
            "Company Type",
            ["mnc","startup","small","mid_size","large_indian"],
            index=2
        )
    
    with col2:
        st.markdown("<div class='section-header'>💰 Financial Information</div>", unsafe_allow_html=True)
        
        monthly_salary = st.number_input("Monthly Salary (₹)", 1000, 500000, 90000, step=1000)
        bank_balance = st.number_input("Bank Balance (₹)", 0, 1000000, 200000, step=5000)
        
        current_emi = st.number_input("Current EMI (₹)", 0, 50000, 5000, step=500)
        credit_score = st.slider("Credit Score", 300, 900, 750, help="Your CIBIL or credit score")
        
        requested_amount = st.number_input("Requested Loan Amount (₹)", 10000, 1000000, 200000, step=5000)
        requested_tenure = st.slider("Loan Tenure (Months)", 6, 60, 36, help="Number of months to repay")
    
    st.markdown("<div class='section-header'>🏠 Monthly Expenses</div>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        monthly_rent = st.number_input("Monthly Rent (₹)", 0, 50000, 0, step=500)
        school_fees = st.number_input("School Fees (₹)", 0, 20000, 2000, step=100)
        college_fees = st.number_input("College Fees (₹)", 0, 20000, 0, step=100)
    
    with col4:
        travel_expenses = st.number_input("Travel Expenses (₹)", 0, 20000, 3000, step=100)
        groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0, 20000, 6000, step=100)
        other_expenses = st.number_input("Other Expenses (₹)", 0, 20000, 4000, step=100)
    
    st.markdown("<div class='section-header'>🔍 Additional Details</div>", unsafe_allow_html=True)
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        family_size = st.selectbox("Family Size", [1,2,3,4,5,6,7,8], index=2)
    
    with col6:
        house_type = st.selectbox("House Type", ["own","rented","family"], index=0)
    
    with col7:
        existing_loans = st.selectbox("Existing Loans", ["yes","no"], index=0)
    
    emi_scenario = st.selectbox(
        "Loan Purpose",
        [
            "vehicle_emi",
            "personal_loan_emi",
            "education_emi",
            "home_appliances_emi",
            "e_commerce_shopping_emi"
        ],
        index=1
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prediction button
    if st.button("🚀 Check Eligibility", use_container_width=True):
        
        # Validation rules
        if age < 21:
            st.error("❌ Minimum age requirement: 21 years")
            st.stop()
        
        if years_of_employment < 1:
            st.error("❌ Minimum employment requirement: 1 year")
            st.stop()
        
        # Create dataframe
        df = pd.DataFrame([{
            'age': age,
            'years_of_employment': years_of_employment,
            'monthly_rent': monthly_rent,
            'family_size': family_size,
            'school_fees': school_fees,
            'college_fees': college_fees,
            'travel_expenses': travel_expenses,
            'groceries_utilities': groceries_utilities,
            'other_monthly_expenses': other_expenses,
            'current_emi_amount': current_emi,
            'credit_score': credit_score,
            'requested_tenure': requested_tenure,
            'monthly_salary': monthly_salary,
            'requested_amount': requested_amount,
            'bank_balance': bank_balance,
            'gender': gender,
            'marital_status': marital_status,
            'education': education,
            'employment_type': employment_type,
            'company_type': company_type,
            'house_type': house_type,
            'existing_loans': existing_loans,
            'emi_scenario': emi_scenario
        }])
        
        # Feature engineering
        df = engineer_features(df)
        df = pd.get_dummies(df)
        
        df_clf = df.reindex(columns=clf_feature_cols, fill_value=0)
        df_reg = df.reindex(columns=reg_feature_cols, fill_value=0)
        
        # Prediction
        pred = clf_model.predict(df_clf)[0]
        
        label_map = {0:"Eligible", 1:"High Risk", 2:"Not Eligible"}
        result = label_map.get(pred)
        
        # Display results
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📊 Prediction Results</div>", unsafe_allow_html=True)
        
        if result == "Eligible":
            st.success("✅ Congratulations! You are ELIGIBLE for the loan")
            
            emi = reg_model.predict(df_reg)[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("💰 Estimated EMI", f"₹{emi:,.0f}")
            
            with col2:
                st.metric("📅 Loan Tenure", f"{requested_tenure} months")
            
            with col3:
                st.metric("💵 Total Amount", f"₹{emi * requested_tenure:,.0f}")
            
            st.markdown("""
            <div class='info-card' style='border-left-color: #10B981;'>
                <h4 style='color: #10B981; margin-top: 0;'>✨ What This Means</h4>
                <p style='color: #555; line-height: 1.6;'>
                    Based on your financial profile, you qualify for the requested loan amount. 
                    Your estimated monthly EMI is shown above. 
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        elif result == "High Risk":
            st.warning("⚠️ Your application is flagged as HIGH RISK")
            
            st.markdown("""
            <div class='info-card' style='border-left-color: #F59E0B;'>
                <h4 style='color: #F59E0B; margin-top: 0;'>⚠️ What This Means</h4>
                <p style='color: #555; line-height: 1.6;'>
                    While you may potentially qualify, your financial profile indicates higher risk factors. 
                    This could be due to credit score, existing EMIs, income or other risk factors.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.error("❌ Unfortunately, you are NOT ELIGIBLE for the loan at this time")
            
            st.markdown("""
            <div class='info-card' style='border-left-color: #EF4444;'>
                <h4 style='color: #EF4444; margin-top: 0;'>🔍 What This Means</h4>
                <p style='color: #555; line-height: 1.6;'>
                    Based on the current parameters, the loan cannot be approved. This could be due to 
                    insufficient income, high existing EMI burden, low credit score, or other risk factors. 
                </p>
            </div>
            """, unsafe_allow_html=True)