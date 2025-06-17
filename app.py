# app.py  —  4‑page Streamlit interface
# -------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Loan Application Approval Predictor",
                   layout="wide",
                   page_icon="🏦")

# ---------------- Load data & constants --------------------
DATA_PATH = Path("loan_applications_data.csv")

MODEL_MAP = {
    "Random Forest": ("loan_approval_rf.pkl", "num_scaler.pkl"),
    "Logistic Regression": ("loan_approval_lr.pkl", "num_scaler.pkl"),
}

@st.cache_resource(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

NUM_COLS = [
    "loan_amount_requested", "loan_tenure_months", "interest_rate_offered",
    "monthly_income", "cibil_score", "existing_emis_monthly",
    "debt_to_income_ratio", "applicant_age", "number_of_dependents"
]
CAT_COLS = [
    "purpose_of_loan", "employment_status", "property_ownership_status",
    "gender", "loan_type"
]
DUMMY_COLS = pd.get_dummies(df[CAT_COLS], drop_first=True).columns

# --------------- Helper functions --------------------------
def load_model_and_scaler(model_key: str):
    model_file, scaler_file = MODEL_MAP[model_key]
    model  = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    return model, scaler

def preprocess(user_dict, scaler):
    """Return a single‑row DF ready for model.predict_proba()."""
    new_df = pd.DataFrame([user_dict])
    num = new_df[NUM_COLS]
    cat = pd.get_dummies(new_df[CAT_COLS], drop_first=True).reindex(
        columns=DUMMY_COLS, fill_value=0)
    X = pd.concat([num, cat], axis=1)
    X[NUM_COLS] = scaler.transform(X[NUM_COLS])
    return X

def applicant_form(container):
    """Render the applicant input widgets inside *container* (main page)."""
    c1, c2 = container.columns(2)
    ui = dict(
        loan_amount_requested = c1.number_input("Loan Amount (₹)", 10000, 1_00_00_000, 2_50_000, 10_000),
        loan_tenure_months    = c2.number_input("Tenure (months)", 6, 360, 60, 6),
        interest_rate_offered = c1.number_input("Interest Rate (%)", 5.0, 30.0, 11.5, 0.1),
        monthly_income        = c2.number_input("Monthly Income (₹)", 5_000, 10_00_000, 90_000, 1_000),
        cibil_score           = c1.number_input("CIBIL Score", 300, 900, 740, 1),
        existing_emis_monthly = c2.number_input("Existing EMIs (₹)", 0, 5_00_000, 5_000, 1_000),
        debt_to_income_ratio  = c1.number_input("Debt‑to‑Income Ratio", 0.0, 1.0, 0.25, 0.01),
        applicant_age         = c2.number_input("Applicant Age", 18, 80, 29, 1),
        number_of_dependents  = c1.number_input("Dependents", 0, 10, 2, 1),

        purpose_of_loan         = c2.selectbox("Purpose of Loan", sorted(df["purpose_of_loan"].unique())),
        employment_status       = c1.selectbox("Employment Status", sorted(df["employment_status"].unique())),
        property_ownership_status = c2.selectbox("Property Ownership", sorted(df["property_ownership_status"].unique())),
        gender                  = c1.selectbox("Gender", sorted(df["gender"].unique())),
        loan_type               = c2.selectbox("Loan Type", sorted(df["loan_type"].unique())),
    )
    return ui

# ---------------- Sidebar navigation -----------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ("🏠 Home", "📁 Dataset", "📊 Visualizations", "🧠 Predictor")
)

# ====================== 🏠 HOME =============================
if page == "🏠 Home":
    st.title("Loan Application Approval Predictor")
    st.header("Welcome to the Loan Application Approval Predictor App")
    st.markdown(
        """
        ### 🔍 What this app offers
        * **View** the dataset used to train the model  
        * **Explore** visual insights such as approval rates and distributions  
        * **Predict** loan approval using:  
          * 🌲 Random Forest  
          * 📈 Logistic Regression
        ---
        Select a page on the left to get started ▶️
        """
    )
    st.markdown("---")
    st.markdown("👨‍💻 Developed by: **M.Pavan Kumar**")

# ================== 📁 DATASET PAGE ========================
elif page == "📁 Dataset":
    st.header("📄 Dataset Preview ")
    st.dataframe(df.head(300), use_container_width=True)

# ================= 📊 VISUALIZATIONS PAGE =================
elif page == "📊 Visualizations":
    st.header("📊 Quick Visualizations")

    # Loan Status Distribution
    status_counts = df["loan_status"].value_counts().reset_index(name="count")
    fig1 = px.bar(status_counts, x="loan_status", y="count",
                  labels={"loan_status": "Loan Status", "count": "Count"},
                  color_discrete_sequence=["royalblue"],
                  title="Loan Status Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    # Approval % by Loan Type
    st.subheader("Approval % by Loan Type")
    plot_df = (
        df.groupby("loan_type")["loan_status"]
          .value_counts(normalize=True)
          .mul(100)
          .rename("percentage")
          .reset_index()
    )
    fig2 = px.bar(plot_df, x="loan_type", y="percentage",
                  color="loan_status", barmode="group",
                  labels={"percentage": "Approval Rate (%)", "loan_type": "Loan Type"})
    st.plotly_chart(fig2, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Heatmap (Numeric Features)")
    corr = df[NUM_COLS].corr()
    fig3 = px.imshow(corr, text_auto=".2f", aspect="auto",
                     color_continuous_scale="RdBu")
    st.plotly_chart(fig3, use_container_width=True)

    # Hist & Boxplot for each numeric feature
    st.subheader("Distribution & Boxplot of Numeric Features")
    for col in NUM_COLS:
        st.markdown(f"**{col.replace('_', ' ').title()}**")
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df[col], kde=True, ax=axs[0], color="skyblue", edgecolor="black")
        axs[0].set_title("Distribution")
        sns.boxplot(y=df[col], ax=axs[1], color="lightcoral")
        axs[1].set_title("Boxplot")
        plt.tight_layout()
        st.pyplot(fig)

# ============= 🧠 MODEL SELECTION & PREDICTION =============
else:
    st.header("🏦 Loan Application Approval Predictor")

    # --- Model chooser in sidebar ---
    model_choice = st.sidebar.selectbox("Choose model", list(MODEL_MAP.keys()))
    model, scaler = load_model_and_scaler(model_choice)

    # --- Applicant form in CENTER ---
    st.subheader("✍️ Enter Applicant Details")
    user_dict = applicant_form(st)   # main page container

    # --- Predict button ---
    if st.button("🚀 Predict"):
        X_ready = preprocess(user_dict, scaler)
        prob = model.predict_proba(X_ready)[0, 1]
        pred = model.predict(X_ready)[0]

        st.subheader("Result")
        if pred == 1:
            st.success(f"✅ Approved  (probability {prob:.1%})")
        else:
            st.error(f"❌ Rejected  (probability {prob:.1%})")
        st.progress(min(int(prob*100), 100))

        with st.expander("See entered details"):
            st.json(user_dict)
    else:
        st.info("Fill the form above and click **Predict**.")
