
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Universal Bank Dashboard", layout="wide")

# Color theme
st.markdown("<h1 style='color:#2E86C1;'>🏦 Universal Bank Loan Intelligence Dashboard</h1>", unsafe_allow_html=True)

# Load data
df = pd.read_csv("UniversalBank.csv")

df_model = df.drop(columns=["ID","ZIPCode"])
X = df_model.drop("Personal Loan", axis=1)
y = df_model["Personal Loan"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Descriptive Analysis","📈 Model Performance","🧠 Insights","📥 Prediction Tool"])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Customer Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Age Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["Age"])
        st.pyplot(fig)
        st.info("Most customers fall between mid-age groups.")

    with col2:
        st.write("### Income Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["Income"])
        st.pyplot(fig)
        st.info("Higher income customers are fewer but important targets.")

    st.write("### Loan Acceptance Distribution")
    fig, ax = plt.subplots()
    df["Personal Loan"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)
    st.info("Only small % accepts loan → need targeted marketing.")

# ---------------- TAB 2 ----------------
with tab2:
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test,y_pred),
            "Precision": precision_score(y_test,y_pred),
            "Recall": recall_score(y_test,y_pred),
            "F1": f1_score(y_test,y_pred)
        })

    res_df = pd.DataFrame(results)
    st.dataframe(res_df)

    st.write("### ROC Curve")
    fig, ax = plt.subplots()

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test,y_prob)
        ax.plot(fpr,tpr,label=name)

    ax.plot([0,1],[0,1],'--')
    ax.legend()
    st.pyplot(fig)

    st.write("### Confusion Matrix (Random Forest)")
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    cm = confusion_matrix(y_test, model.predict(X_test))

    fig, ax = plt.subplots()
    ax.imshow(cm)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    st.pyplot(fig)

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Business Insights")

    st.success("High income + high credit usage customers are more likely to accept loans.")
    st.success("Customers with CD accounts show strong conversion potential.")
    st.success("Focus on digital users → lower cost marketing.")

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("Upload Data for Prediction")

    file = st.file_uploader("Upload CSV")

    if file:
        test = pd.read_csv(file)
        model = RandomForestClassifier()
        model.fit(X_train,y_train)

        preds = model.predict(test)
        probs = model.predict_proba(test)[:,1]

        test["Prediction"] = preds
        test["Probability"] = probs

        st.dataframe(test)

        st.download_button("Download Results", test.to_csv(index=False), "predictions.csv")
