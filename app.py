import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import silhouette_score
st.set_page_config(page_icon = '🛃', page_title="Customer Segmentation",  layout="wide")
file = st.file_uploader("Mall_Customers.csv", type=["csv"])
df = None
if file:
    df = pd.read_csv(file)
with st.sidebar:
    st.title("Customer Segmentation")
    if df is not None:
        features = st.multiselect("Select Features: ",options=df.columns , default = ["Annual Income (k$)","Spending Score (1-100)"]) 
        df = df.loc[:,features]
def preprocess_data(df):
    encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = encoder.fit_transform(df[col])
    return df
def elbow():
    out = []
    k_values = range(1,11)
    for i in k_values:
        model = KMeans(n_clusters = i)
        model.fit(df)
        out.append(model.inertia_)
    KL = KneeLocator(k_values, out, curve="convex", direction="decreasing")
    df1 = pd.DataFrame({"k_values":k_values, "inertia":out})
    st.subheader("ELBOW CURVE")
    fig = st.line_chart(data = df1, x = "k_values", y = "inertia")
    return KL.elbow
if df is not None:
    st.subheader("Sample Data")
    st.write(df.sample(10))
    df = preprocess_data(df)
    k = elbow()
    model = KMeans(n_clusters = k)
    model.fit(df)
    labels = model.labels_
    df["clusters"] = labels
    st.subheader("Clustered Data")
    st.scatter_chart(data = df, x = 'Annual Income (k$)', y = 'Spending Score (1-100)', color = "clusters")