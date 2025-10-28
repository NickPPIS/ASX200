import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
import plotly.express as px

st.set_page_config(page_title="ASX200 Clustering", layout="wide")

# ---------------------------------
# Load and prepare the data
# ---------------------------------
@st.cache_data
def load_data():
    # Load columns F:H, K:M (adding column M for Volume)
    df = pd.read_excel("ASX200_list.xlsx", sheet_name="Sheet2", usecols="F:H,K:M")
    df.columns = ["Ticker", "Date", "Price", "Security", "MarketCap", "Volume"]
    
    # Forward-fill missing tickers
    df["Ticker"] = df["Ticker"].ffill()

    # Remove rows with no Price or MarketCap
    df = df.dropna(subset=["Price", "MarketCap"])

    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # Calculate daily returns per Ticker
    df = df.sort_values(by=["Ticker", "Date"])
    df["DailyReturn"] = df.groupby("Ticker")["Price"].pct_change()

    # Aggregate to one row per ticker (average daily return + last market cap + last volume)
    df_summary = (
        df.groupby(["Ticker", "Security"], as_index=False)
        .agg({"MarketCap": "last", "DailyReturn": "mean", "Volume": "last"})
        .dropna()
    )

    return df_summary

# Load data
df = load_data()

# ---------------------------------
# User selections
# ---------------------------------
axis_options = ["MarketCap", "DailyReturn", "Volume"]

x_axis = st.selectbox("Select X-axis variable:", axis_options)
y_axis = st.selectbox("Select Y-axis variable:", axis_options)

clustering_method = st.selectbox(
    "Choose clustering method:",
    ["KMeans", "Agglomerative", "DBSCAN", "MeanShift"]
)

# ---------------------------------
# Scale and cluster
# ---------------------------------
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[[x_axis, y_axis]])

if clustering_method == "KMeans":
    k = st.slider("Number of clusters (k)", 2, 10, 3)
    model = KMeans(n_clusters=k, random_state=42)
elif clustering_method == "Agglomerative":
    k = st.slider("Number of clusters (k)", 2, 10, 3)
    model = AgglomerativeClustering(n_clusters=k)
elif clustering_method == "DBSCAN":
    eps = st.slider("Epsilon", 0.1, 5.0, 1.0)
    min_samples = st.slider("Min Samples", 1, 10, 2)
    model = DBSCAN(eps=eps, min_samples=min_samples)
else:
    model = MeanShift()

df["Cluster"] = model.fit_predict(df_scaled)

# ---------------------------------
# Display results
# ---------------------------------

# Show clustered data table WITHOUT Ticker column
st.write("### Clustered Data")
st.dataframe(df.drop(columns=["Ticker"]))

# Plot scatter using Plotly WITHOUT ticker
fig = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color=df["Cluster"].astype(str),  # Convert to string for categorical coloring
    hover_data=["Security", "MarketCap", "DailyReturn", "Volume"],  # Ticker removed
    title=f"{clustering_method} Clusters: {x_axis} vs {y_axis}"
)

fig.update_layout(legend_title_text='Cluster')

st.plotly_chart(fig, use_container_width=True)
