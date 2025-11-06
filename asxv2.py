import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import plotly.express as px
import streamlit as st

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="ASX 200 Clustering Dashboard", layout="wide")
st.title("ASX 200 Stocks Clustering Dashboard")

# -----------------------------
# Load Excel Data (Relative Path)
# -----------------------------
# Get the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct a relative path to the Excel file
file_path = os.path.join(base_dir, "ASX200_list.xlsx")

try:
    df = pd.read_excel(file_path, sheet_name="Sheet2")
except FileNotFoundError:
    st.error(
        f"❌ File not found: {file_path}\n\n"
        "Make sure 'ASX200_list.xlsx' is located in the same folder as 'asxv2.py' "
        "and has been pushed to your GitHub repository before deploying to Streamlit Cloud."
    )
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading Excel file: {e}")
    st.stop()

# Clean up column names
df.columns = df.columns.str.strip()

# -----------------------------
# Check for required columns
# -----------------------------
required_cols = ['Security', 'MarketCap', 'Avg Daily Return', 'Avg Daily Vol']
missing = [col for col in required_cols if col not in df.columns]

if missing:
    st.error(f"Missing columns in Excel sheet: {missing}")
    st.stop()

# Keep only relevant columns and drop missing values
df = df[required_cols].dropna()

# -----------------------------
# Feature Scaling
# -----------------------------
features = ['MarketCap', 'Avg Daily Return', 'Avg Daily Vol']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Clustering Settings")

cluster_method = st.sidebar.selectbox(
    "Select clustering method",
    ["KMeans", "Agglomerative", "DBSCAN"]
)

if cluster_method in ["KMeans", "Agglomerative"]:
    num_clusters = st.sidebar.slider("Number of clusters", 2, 10, 5)
else:
    num_clusters = None

if cluster_method == "DBSCAN":
    eps_val = st.sidebar.slider("DBSCAN: eps (radius)", 0.1, 5.0, 1.0, 0.1)
    min_samples_val = st.sidebar.slider("DBSCAN: min_samples", 2, 20, 5, 1)

# -----------------------------
# Apply Clustering
# -----------------------------
if cluster_method == "KMeans":
    model = KMeans(n_clusters=num_clusters, random_state=42)
    df["Cluster"] = model.fit_predict(X_scaled)

elif cluster_method == "Agglomerative":
    model = AgglomerativeClustering(n_clusters=num_clusters)
    df["Cluster"] = model.fit_predict(X_scaled)

elif cluster_method == "DBSCAN":
    model = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    df["Cluster"] = model.fit_predict(X_scaled)
    df["Cluster"] = df["Cluster"].astype(str)
    df.loc[df["Cluster"] == "-1", "Cluster"] = "Noise"

# Convert to string for consistent legend display
df["Cluster"] = df["Cluster"].astype(str)

# -----------------------------
# X-Y Variable Selection
# -----------------------------
st.sidebar.header("Chart Axes")
x_axis = st.sidebar.selectbox("Select X-axis variable", features, index=2)
y_axis = st.sidebar.selectbox("Select Y-axis variable", features, index=1)

# -----------------------------
# Interactive Scatter Plot
# -----------------------------
bright_palette = px.colors.qualitative.Vivid

fig = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    size="MarketCap",
    color="Cluster",
    color_discrete_sequence=bright_palette,
    hover_data=["Security"],
    title=f"{cluster_method} Clusters: {y_axis} vs {x_axis}",
    height=650
)

fig.update_xaxes(tickformat=",")
fig.update_yaxes(tickformat=",")

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Cluster Summary Table
# -----------------------------
st.subheader("Cluster Summary")

if cluster_method != "DBSCAN":
    cluster_summary = df.groupby("Cluster")[features].mean().reset_index()
    st.dataframe(cluster_summary)
else:
    st.info("Cluster summary not available for DBSCAN (variable cluster count).")

# -----------------------------
# View Stocks in Each Cluster
# -----------------------------
st.sidebar.header("Cluster Explorer")

if cluster_method != "DBSCAN":
    selected_cluster = st.sidebar.selectbox(
        "View stocks in cluster",
        sorted(df["Cluster"].unique())
    )
    st.subheader(f"Stocks in Cluster {selected_cluster}")
    st.dataframe(df[df["Cluster"] == selected_cluster][["Security"] + features])
else:
    st.subheader("DBSCAN Cluster Breakdown")
    st.dataframe(df[["Security", "Cluster"] + features])
