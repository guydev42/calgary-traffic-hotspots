"""
Calgary Traffic Incident Hotspot Analyzer - Streamlit Application.

Interactive dashboard for exploring traffic incident patterns, spatial
hotspot clusters, temporal trends, and classification model performance
using data from Calgary Open Data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_and_prepare_data,
    create_clustering_features,
    create_classification_features,
)
from src.model import SpatialClusterAnalyzer, IncidentClassifier

# ---------------------------------------------------------------------------
# Color palette (blues / purples consistent with portfolio)
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#4A90D9",
    "secondary": "#7B68EE",
    "accent": "#6C5CE7",
    "dark": "#2D3436",
    "light": "#DFE6E9",
    "success": "#00B894",
    "warning": "#FDCB6E",
    "danger": "#D63031",
}

COLOR_SEQUENCE = [
    "#4A90D9",
    "#7B68EE",
    "#6C5CE7",
    "#00B894",
    "#FDCB6E",
    "#E17055",
    "#00CEC9",
    "#D63031",
    "#636E72",
    "#A29BFE",
]

PLOTLY_TEMPLATE = "plotly_white"

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Calgary Traffic Incident Hotspot Analyzer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4A90D9;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #636E72;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #4A90D9, #7B68EE);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.85;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading with caching
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading traffic incident data...")
def load_data() -> pd.DataFrame:
    """Load and preprocess traffic incident data with caching."""
    return load_and_prepare_data(limit=100000)


@st.cache_resource(show_spinner="Running spatial clustering...")
def run_clustering(
    _coords: np.ndarray,
    n_kmeans: int = 8,
    dbscan_eps: float = 0.005,
    dbscan_min: int = 10,
):
    """Run DBSCAN and KMeans clustering (cached as a resource)."""
    analyzer = SpatialClusterAnalyzer()
    dbscan_labels = analyzer.fit_dbscan(
        _coords, eps=dbscan_eps, min_samples=dbscan_min
    )
    kmeans_labels = analyzer.fit_kmeans(_coords, n_clusters=n_kmeans)
    return analyzer, dbscan_labels, kmeans_labels


@st.cache_resource(show_spinner="Training classification models...")
def run_classification(_X: pd.DataFrame, _y: pd.Series):
    """Train and evaluate classifiers (cached as a resource)."""
    classifier = IncidentClassifier()
    results = classifier.train_and_evaluate(_X, _y)
    return classifier, results


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<p style="font-size:1.4rem; font-weight:700; color:#4A90D9;">'
    "Navigation</p>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Go to",
    [
        "Incident Dashboard",
        "Hotspot Map",
        "Temporal Analysis",
        "Model Performance",
        "About",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data Source:** [Calgary Open Data]"
    "(https://data.calgary.ca/Transportation-Transit/Traffic-Incidents/35ra-9556)"
)
st.sidebar.markdown("**Dataset ID:** `35ra-9556`")
st.sidebar.markdown("**Updated:** Every 10 minutes")

# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------
try:
    df = load_data()
    data_loaded = True
except Exception as exc:
    st.error(f"Failed to load data: {exc}")
    st.info(
        "Please ensure you have internet access or a cached CSV in the data/ folder."
    )
    data_loaded = False
    df = pd.DataFrame()


# ===================================================================
# PAGE: Incident Dashboard
# ===================================================================
if page == "Incident Dashboard" and data_loaded:
    st.markdown(
        '<p class="main-header">Incident Dashboard</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">'
        "Overview of Calgary traffic incidents sourced from Calgary Open Data."
        "</p>",
        unsafe_allow_html=True,
    )

    # --- Key metrics ---------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    total_incidents = len(df)
    unique_quadrants = df["quadrant"].nunique() if "quadrant" in df.columns else 0

    if "start_dt" in df.columns and pd.api.types.is_datetime64_any_dtype(
        df["start_dt"]
    ):
        min_date = df["start_dt"].min().strftime("%Y-%m-%d")
        max_date = df["start_dt"].max().strftime("%Y-%m-%d")
    else:
        min_date = "N/A"
        max_date = "N/A"

    with col1:
        st.metric("Total Incidents", f"{total_incidents:,}")
    with col2:
        st.metric("Quadrants Covered", unique_quadrants)
    with col3:
        st.metric("Earliest Record", min_date)
    with col4:
        st.metric("Latest Record", max_date)

    st.markdown("---")

    # --- Incidents by quadrant -----------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Incidents by Quadrant")
        if "quadrant" in df.columns:
            quad_counts = (
                df["quadrant"]
                .value_counts()
                .reset_index()
                .rename(columns={"index": "quadrant", "quadrant": "count"})
            )
            # Handle pandas version differences
            if "count" not in quad_counts.columns:
                quad_counts.columns = ["quadrant", "count"]
            fig_quad = px.bar(
                quad_counts,
                x="quadrant",
                y="count",
                color="quadrant",
                color_discrete_sequence=COLOR_SEQUENCE,
                template=PLOTLY_TEMPLATE,
                labels={"count": "Number of Incidents", "quadrant": "Quadrant"},
            )
            fig_quad.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_quad, use_container_width=True)
        else:
            st.warning("Quadrant data not available.")

    with col_right:
        st.subheader("Incident Description Analysis")
        desc_col = None
        for candidate in ["description", "DESCRIPTION", "incident_info"]:
            if candidate in df.columns:
                desc_col = candidate
                break

        if desc_col is not None:
            desc_counts = (
                df[desc_col]
                .value_counts()
                .head(10)
                .reset_index()
            )
            desc_counts.columns = ["description", "count"]
            fig_desc = px.bar(
                desc_counts,
                x="count",
                y="description",
                orientation="h",
                color="count",
                color_continuous_scale=["#7B68EE", "#4A90D9"],
                template=PLOTLY_TEMPLATE,
                labels={"count": "Occurrences", "description": "Incident Type"},
            )
            fig_desc.update_layout(
                showlegend=False,
                height=400,
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig_desc, use_container_width=True)
        else:
            st.info(
                "No description column found. Available columns: "
                + ", ".join(df.columns[:15].tolist())
            )

    # --- Data sample ---------------------------------------------------
    with st.expander("View Raw Data Sample"):
        st.dataframe(df.head(100), use_container_width=True)

# ===================================================================
# PAGE: Hotspot Map
# ===================================================================
elif page == "Hotspot Map" and data_loaded:
    st.markdown(
        '<p class="main-header">Traffic Incident Hotspot Map</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">'
        "Spatial clusters of traffic incidents identified via DBSCAN and KMeans."
        "</p>",
        unsafe_allow_html=True,
    )

    # --- Clustering parameters -----------------------------------------
    with st.sidebar:
        st.markdown("### Clustering Settings")
        cluster_method = st.selectbox(
            "Clustering Method", ["KMeans", "DBSCAN"]
        )
        if cluster_method == "KMeans":
            n_clusters = st.slider(
                "Number of clusters", min_value=3, max_value=20, value=8
            )
        else:
            dbscan_eps = st.slider(
                "DBSCAN eps (degrees)",
                min_value=0.001,
                max_value=0.05,
                value=0.005,
                step=0.001,
                format="%.3f",
            )
            dbscan_min = st.slider(
                "DBSCAN min_samples",
                min_value=3,
                max_value=50,
                value=10,
            )

        sample_size = st.slider(
            "Map sample size",
            min_value=1000,
            max_value=min(len(df), 50000),
            value=min(10000, len(df)),
            step=1000,
        )

    # --- Subsample for performance -------------------------------------
    df_map = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
    coords = create_clustering_features(df_map)

    if len(coords) < 10:
        st.warning("Not enough valid coordinates for clustering.")
    else:
        if cluster_method == "KMeans":
            analyzer, _, kmeans_labels = run_clustering(
                coords, n_kmeans=n_clusters
            )
            labels = kmeans_labels
        else:
            analyzer, dbscan_labels, _ = run_clustering(
                coords, dbscan_eps=dbscan_eps, dbscan_min=dbscan_min
            )
            labels = dbscan_labels

        df_map = df_map.iloc[: len(labels)].copy()
        df_map["cluster"] = labels.astype(str)

        # --- Scatter Mapbox -------------------------------------------
        fig_map = px.scatter_mapbox(
            df_map,
            lat="latitude",
            lon="longitude",
            color="cluster",
            color_discrete_sequence=COLOR_SEQUENCE,
            zoom=10,
            center={"lat": 51.05, "lon": -114.07},
            mapbox_style="carto-positron",
            opacity=0.6,
            hover_data={
                "latitude": ":.4f",
                "longitude": ":.4f",
                "cluster": True,
            },
            title=f"Traffic Incidents Colored by {cluster_method} Cluster",
        )
        fig_map.update_layout(
            height=650,
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # --- Cluster summary ------------------------------------------
        st.subheader("Cluster Summary")
        summary = analyzer.get_cluster_summary(df_map, labels)
        st.dataframe(summary, use_container_width=True)

# ===================================================================
# PAGE: Temporal Analysis
# ===================================================================
elif page == "Temporal Analysis" and data_loaded:
    st.markdown(
        '<p class="main-header">Temporal Analysis</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">'
        "Explore when traffic incidents occur by hour, day, and month."
        "</p>",
        unsafe_allow_html=True,
    )

    # --- Incidents by Hour of Day -------------------------------------
    if "hour" in df.columns:
        st.subheader("Incidents by Hour of Day")
        hourly = df.groupby("hour").size().reset_index(name="count")
        fig_hour = px.bar(
            hourly,
            x="hour",
            y="count",
            color="count",
            color_continuous_scale=["#7B68EE", "#4A90D9", "#D63031"],
            template=PLOTLY_TEMPLATE,
            labels={"hour": "Hour of Day (0-23)", "count": "Incident Count"},
        )
        fig_hour.update_layout(height=400)
        st.plotly_chart(fig_hour, use_container_width=True)

    # --- Incidents by Day of Week -------------------------------------
    if "day_of_week" in df.columns:
        st.subheader("Incidents by Day of Week")
        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        if "day_name" in df.columns:
            daily = df["day_name"].value_counts().reindex(day_order).reset_index()
            daily.columns = ["day", "count"]
        else:
            daily = df.groupby("day_of_week").size().reset_index(name="count")
            daily["day"] = daily["day_of_week"].map(
                {i: d for i, d in enumerate(day_order)}
            )

        fig_dow = px.bar(
            daily,
            x="day",
            y="count",
            color="count",
            color_continuous_scale=["#6C5CE7", "#4A90D9"],
            template=PLOTLY_TEMPLATE,
            labels={"day": "Day of Week", "count": "Incident Count"},
        )
        fig_dow.update_layout(height=400)
        st.plotly_chart(fig_dow, use_container_width=True)

    # --- Incidents by Month -------------------------------------------
    if "month" in df.columns:
        st.subheader("Incidents by Month")
        monthly = df.groupby("month").size().reset_index(name="count")
        month_names = [
            "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        monthly["month_name"] = monthly["month"].apply(
            lambda m: month_names[int(m)] if 1 <= int(m) <= 12 else str(m)
        )
        fig_month = px.bar(
            monthly,
            x="month_name",
            y="count",
            color="count",
            color_continuous_scale=["#4A90D9", "#7B68EE"],
            template=PLOTLY_TEMPLATE,
            labels={"month_name": "Month", "count": "Incident Count"},
        )
        fig_month.update_layout(height=400)
        st.plotly_chart(fig_month, use_container_width=True)

    # --- Year-over-Year Trend -----------------------------------------
    if "year" in df.columns:
        st.subheader("Year-over-Year Trend")
        yearly = df.groupby("year").size().reset_index(name="count")
        yearly = yearly[yearly["year"] > 2000]  # Filter out bad year values
        fig_year = px.line(
            yearly,
            x="year",
            y="count",
            markers=True,
            template=PLOTLY_TEMPLATE,
            labels={"year": "Year", "count": "Incident Count"},
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig_year.update_layout(height=400)
        st.plotly_chart(fig_year, use_container_width=True)

    # --- Heatmap: Hour x Day of Week ----------------------------------
    if "hour" in df.columns and "day_of_week" in df.columns:
        st.subheader("Incident Heatmap: Hour vs Day of Week")
        heatmap_data = (
            df.groupby(["day_of_week", "hour"])
            .size()
            .reset_index(name="count")
            .pivot(index="day_of_week", columns="hour", values="count")
            .fillna(0)
        )
        day_labels = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        heatmap_data.index = [
            day_labels[i] if i < len(day_labels) else str(i)
            for i in heatmap_data.index
        ]

        fig_heatmap = px.imshow(
            heatmap_data,
            aspect="auto",
            color_continuous_scale=["#DFE6E9", "#7B68EE", "#D63031"],
            template=PLOTLY_TEMPLATE,
            labels={"x": "Hour of Day", "y": "Day of Week", "color": "Incidents"},
        )
        fig_heatmap.update_layout(height=450)
        st.plotly_chart(fig_heatmap, use_container_width=True)

# ===================================================================
# PAGE: Model Performance
# ===================================================================
elif page == "Model Performance" and data_loaded:
    st.markdown(
        '<p class="main-header">Model Performance</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">'
        "Classification models predicting high-incident hours and locations."
        "</p>",
        unsafe_allow_html=True,
    )

    # --- Train models --------------------------------------------------
    X, y = create_classification_features(df)
    if X.empty or y is None:
        st.error("Could not create classification features from the data.")
    else:
        classifier, results = run_classification(X, y)

        # --- Results comparison table ---------------------------------
        st.subheader("Classifier Comparison")
        results_df = classifier.get_results_dataframe()
        if not results_df.empty:
            st.dataframe(
                results_df.style.format(
                    {
                        "accuracy": "{:.4f}",
                        "precision": "{:.4f}",
                        "recall": "{:.4f}",
                        "f1_score": "{:.4f}",
                        "cv_f1_mean": "{:.4f}",
                        "cv_f1_std": "{:.4f}",
                    }
                ),
                use_container_width=True,
            )

            # --- Metric comparison bar chart --------------------------
            metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
            fig_compare = go.Figure()
            for metric in metrics_to_plot:
                fig_compare.add_trace(
                    go.Bar(
                        name=metric.replace("_", " ").title(),
                        x=results_df["model"],
                        y=results_df[metric],
                        text=results_df[metric].apply(lambda v: f"{v:.3f}"),
                        textposition="outside",
                    )
                )
            fig_compare.update_layout(
                barmode="group",
                template=PLOTLY_TEMPLATE,
                height=450,
                title="Model Metrics Comparison",
                yaxis_title="Score",
                colorway=COLOR_SEQUENCE,
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        # --- Feature importance ----------------------------------------
        st.subheader("Feature Importance")
        model_choice = st.selectbox(
            "Select model for feature importance",
            list(classifier.models.keys()),
        )
        importance_df = classifier.get_feature_importance(model_choice)
        if importance_df is not None:
            fig_imp = px.bar(
                importance_df,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=["#7B68EE", "#4A90D9"],
                template=PLOTLY_TEMPLATE,
                labels={
                    "importance": "Importance Score",
                    "feature": "Feature",
                },
                title=f"Feature Importance - {model_choice}",
            )
            fig_imp.update_layout(
                height=500,
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance not available for this model.")

        # --- Classification report ------------------------------------
        st.subheader("Classification Report")
        report = classifier.get_classification_report(model_choice)
        if report:
            st.code(report, language="text")

        # --- Confusion matrix -----------------------------------------
        st.subheader("Confusion Matrix")
        cm = classifier.get_confusion_matrix(model_choice)
        if cm is not None:
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale=["#DFE6E9", "#4A90D9", "#6C5CE7"],
                x=["Predicted Low", "Predicted High"],
                y=["Actual Low", "Actual High"],
                template=PLOTLY_TEMPLATE,
                title=f"Confusion Matrix - {model_choice}",
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)

# ===================================================================
# PAGE: About
# ===================================================================
elif page == "About":
    st.markdown(
        '<p class="main-header">About This Project</p>', unsafe_allow_html=True
    )

    st.markdown(
        """
        ### Project Overview

        The **Calgary Traffic Incident Hotspot Analyzer** is an interactive
        data science application that identifies spatial and temporal patterns
        in traffic incidents across Calgary. By combining unsupervised
        clustering with supervised classification, the tool highlights
        where and when incidents are most likely to occur.

        ### Methodology

        **Spatial Clustering (Unsupervised)**
        - **DBSCAN** (Density-Based Spatial Clustering of Applications with
          Noise): Groups incidents into dense regions using the haversine
          distance metric. Well suited for identifying irregularly shaped
          hotspots and automatically filtering noise points.
        - **KMeans**: Partitions incidents into a user-specified number of
          clusters based on centroid distance. Provides balanced cluster
          sizes and clear geographic zones.

        **Temporal Classification (Supervised)**
        - **Random Forest**: Ensemble of decision trees with feature
          importance extraction. Robust to overfitting with tuned
          hyperparameters.
        - **Gradient Boosting**: Sequential ensemble method that
          iteratively corrects prediction errors. Strong performance
          on structured tabular data.

        ### Features

        | Feature | Description |
        |---|---|
        | **Incident Dashboard** | Key metrics, quadrant breakdown, description analysis |
        | **Hotspot Map** | Interactive Mapbox scatter plot with cluster coloring |
        | **Temporal Analysis** | Hour, day, month, year trends and a heatmap |
        | **Model Performance** | Classifier comparison, feature importance, confusion matrix |

        ### Data Source

        - **Dataset**: Traffic Incidents
        - **Portal**: [Calgary Open Data](https://data.calgary.ca/)
        - **ID**: `35ra-9556`
        - **Records**: 59,000+ (updated every 10 minutes)
        - **Columns**: 13 fields including location, description, timestamps

        ### Technology Stack

        - **Python** (pandas, NumPy, scikit-learn)
        - **Plotly** for interactive visualizations
        - **Streamlit** for the web application
        - **sodapy** for Socrata Open Data API access
        - **joblib** for model persistence

        ### Author

        Built as part of the **Calgary Data Portfolio** project demonstrating
        data science and machine learning capabilities using City of Calgary
        open data.
        """
    )

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#636E72;">'
        "Calgary Traffic Incident Hotspot Analyzer | Calgary Data Portfolio"
        "</p>",
        unsafe_allow_html=True,
    )
