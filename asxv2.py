import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import plotly.express as px
import streamlit as st

# ------------------------------------------------
# Streamlit Page Setup
# ------------------------------------------------
st.set_page_config(page_title="ASX 200 Clustering Dashboard", layout="wide")
st.title("ASX 200 Stocks Clustering Dashboard")

# ------------------------------------------------
# Load Excel Data (Relative Path)
# ------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "ASX200_list.xlsx")

# Load Sheet2 (Main Stock Data)
try:
    df = pd.read_excel(file_path, sheet_name="Sheet2")
except Exception as e:
    st.error(f"❌ Error loading Sheet2: {e}")
    st.stop()

df.columns = df.columns.str.strip()

# ------------------------------------------------
# Load Sheet3 (Sector data)
# ------------------------------------------------
try:
    df_sector = pd.read_excel(file_path, sheet_name="Sheet3")
except Exception as e:
    st.error(f"❌ Error loading Sheet3: {e}")
    st.stop()

df_sector.columns = df_sector.columns.str.strip()

# Check required sector columns
required_sector_cols = ["Security", "Sector"]
missing_sec = [c for c in required_sector_cols if c not in df_sector.columns]

if missing_sec:
    st.error(f"Missing required columns in Sheet3: {missing_sec}")
    st.stop()

# ------------------------------------------------
# Merge Sector Data Into Main DF
# ------------------------------------------------
df = df.merge(df_sector[["Security", "Sector"]], on="Security", how="left")

# ------------------------------------------------
# Validate Required Columns
# ------------------------------------------------
required_cols = ['Security', 'MarketCap', 'Avg Daily Return', 'Avg Daily Vol']
missing = [col for col in required_cols if col not in df.columns]

if missing:
    st.error(f"Missing columns in Excel sheet: {missing}")
    st.stop()

# Drop rows with missing required variables
df = df[required_cols + ["Sector"]].dropna(subset=['MarketCap', 'Avg Daily Return', 'Avg Daily Vol'])

# ------------------------------------------------
# Clean & One-Hot Encode Sector
# ------------------------------------------------
df["Sector"] = df["Sector"].replace(
    ["#VALUE!", "Not Applicable", "", " ", None],
    "Unknown"
)

sector_dummies = pd.get_dummies(df["Sector"], prefix="Sector")
df = pd.concat([df, sector_dummies], axis=1)

# ------------------------------------------------
# Feature Scaling
# ------------------------------------------------
sector_features = list(sector_dummies.columns)
features = ['MarketCap', 'Avg Daily Return', 'Avg Daily Vol'] + sector_features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# ------------------------------------------------
# Sidebar Controls
# ------------------------------------------------
st.sidebar.header("Clustering Settings")

cluster_method = st.sidebar.selectbox(
    "Select clustering method",
    ["KMeans", "Agglomerative", "DBSCAN"]
)

if cluster_method in ["KMeans", "Agglomerative"]:
    num_clusters = st.sidebar.slider("Number of clusters", 2, 10, 4)
else:
    num_clusters = None

if cluster_method == "DBSCAN":
    eps_val = st.sidebar.slider("DBSCAN: eps (radius)", 0.1, 5.0, 1.0, 0.1)
    min_samples_val = st.sidebar.slider("DBSCAN: min_samples", 2, 20, 5, 1)

# ------------------------------------------------
# Apply Clustering
# ------------------------------------------------
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

df["Cluster"] = df["Cluster"].astype(str)

# ------------------------------------------------
# X-Y Axis Variable Selection
# ------------------------------------------------
st.sidebar.header("Chart Axes")

x_axis = st.sidebar.selectbox("Select X-axis variable", features, index=0)
y_axis = st.sidebar.selectbox("Select Y-axis variable", features, index=1)

# ------------------------------------------------
# Chart Data Filtering for Sector Dummies
# ------------------------------------------------
df_plot = df.copy()

# If X or Y is a one-hot sector, only show TRUE = 1 values
if x_axis.startswith("Sector_"):
    df_plot = df_plot[df_plot[x_axis] == 1]

if y_axis.startswith("Sector_"):
    df_plot = df_plot[df_plot[y_axis] == 1]

# Ensure df_plot isn't empty
if df_plot.empty:
    st.warning("No tickers match the selected filters/axes.")
    st.stop()

# ------------------------------------------------
# Interactive Scatter Plot
# ------------------------------------------------
bright_palette = px.colors.qualitative.Vivid

fig = px.scatter(
    df_plot,
    x=x_axis,
    y=y_axis,
    size="MarketCap",
    color="Cluster",
    color_discrete_sequence=bright_palette,
    hover_data=["Security", "Sector"],
    title=f"{cluster_method} Clusters: {y_axis} vs {x_axis}",
    height=650
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# Cluster Summary Table
# ------------------------------------------------
st.subheader("Cluster Summary")

base_features = ['MarketCap', 'Avg Daily Return', 'Avg Daily Vol']

if cluster_method != "DBSCAN":
    cluster_summary = df.groupby("Cluster")[base_features].mean().reset_index()
    st.dataframe(cluster_summary)
else:
    st.info("Cluster summary not available for DBSCAN (variable cluster count).")

# ------------------------------------------------
# Cluster Explorer
# ------------------------------------------------
st.sidebar.header("Cluster Explorer")

if cluster_method != "DBSCAN":
    selected_cluster = st.sidebar.selectbox(
        "View stocks in cluster",
        sorted(df["Cluster"].unique())
    )
    st.subheader(f"Stocks in Cluster {selected_cluster}")
    st.dataframe(df[df["Cluster"] == selected_cluster][["Security", "Sector"] + base_features])
else:
    st.subheader("DBSCAN Cluster Breakdown")
    st.dataframe(df[["Security", "Sector", "Cluster"] + base_features])
