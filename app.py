import streamlit as st
from utils.data_loader import load_sample_regression
from utils.model_simulator import simulate_model

st.set_page_config(
    page_title="ML Model Simulator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ML Model Simulator")
st.write("""
Welcome to the **ML Model Simulator**!  
- Explore ML models like Linear Regression, Logistic Regression, etc.  
- Adjust parameters, visualize results, and test with sample inputs.  
- Navigate using the sidebar.  
""")

# -------------------------
# Dataset Selection Section
# -------------------------
st.sidebar.header("Dataset Selection")

dataset = st.sidebar.selectbox(
    "Choose Dataset",
    ["Iris", "Wine", "Digits", "Sample Regression"]
)

if dataset == "Sample Regression":
    df = load_sample_regression()
    st.subheader("📊 Sample Regression Dataset (first 10 rows)")
    st.dataframe(df.head(10))

    # Optional quick visualization
    st.scatter_chart(df, x="Feature", y="Target")

    # Example of using simulate_model later
    # X = df[["Feature"]].values
    # y = df["Target"].values
    # simulate_model("Linear Regression", X, y)