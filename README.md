<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=220&section=header&text=Traffic%20Incident%20Hotspot%20Analyzer&fontSize=36&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Spatial%20clustering%20and%20temporal%20classification%20on%2060K%2B%20incidents&descSize=16&descAlignY=55&descColor=c8e0ff" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/DBSCAN-Spatial_Clustering-blue?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Accuracy-80%25-228B22?style=for-the-badge" />
  <img src="https://img.shields.io/badge/scikit--learn-ML_Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Calgary_Open_Data-Socrata_API-orange?style=for-the-badge" />
</p>

---

## Table of contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [Dataset](#dataset)
- [Tech stack](#tech-stack)
- [Methodology](#methodology)
- [Acknowledgements](#acknowledgements)

---

## Overview

> **Problem** -- Traffic incidents in Calgary cause delays, economic losses, and safety hazards. Understanding where and when incidents cluster is essential for city planners and emergency services.
>
> **Solution** -- This project applies spatial clustering (DBSCAN with haversine distance) and temporal classification to 60,000+ traffic incidents to identify high-risk hotspots and predict peak-incident periods.
>
> **Impact** -- Enables targeted infrastructure improvements and optimized emergency response deployment by revealing spatial and temporal incident patterns.

---

## Results

| Metric | Value |
|--------|-------|
| Best model | Gradient Boosting |
| Accuracy | ~0.80 |
| CV F1 (5-fold) | ~0.77 |

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌────────────────┐     ┌─────────────────┐
│  Calgary Open   │────>│  Incident data   │────>│  Spatial:        │────>│  Temporal:      │────>│  Streamlit      │
│  Data (Socrata) │     │  Geocoding &     │     │  DBSCAN +        │     │  Gradient       │     │  dashboard      │
│  60K+ incidents │     │  cleaning        │     │  KMeans          │     │  Boosting       │     │  Hotspot map    │
│  Dataset 35ra   │     │  Time parsing    │     │  Haversine dist  │     │  Rush-hour pred │     │  Time analysis  │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └────────────────┘     └─────────────────┘
```

---

## Project structure

<details>
<summary>Click to expand</summary>

```
project_03_traffic_incident_hotspots/
├── app.py                          # Streamlit dashboard
├── index.html                      # Static landing page
├── requirements.txt                # Python dependencies
├── README.md
├── data/
│   └── traffic_incidents.csv       # Cached incident data
├── models/                         # Saved model artifacts
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory data analysis
└── src/
    ├── __init__.py
    ├── data_loader.py              # Data fetching & preprocessing
    └── model.py                    # Clustering & classification
```

</details>

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/guydev42/traffic-incident-hotspots.git
cd traffic-incident-hotspots

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Calgary Open Data -- Traffic Incidents](https://data.calgary.ca/) (dataset `35ra-9556`) |
| Records | 60,000+ |
| Access method | Socrata API (sodapy) |
| Key fields | Latitude, longitude, incident type, timestamp, description |
| Target variable | Peak-incident period (binary classification) |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/sodapy-Socrata_API-orange?style=flat-square" />
</p>

---

## Methodology

### Data ingestion and cleaning

- Fetched real-time traffic incident data from Calgary Open Data (dataset `35ra-9556`)
- Parsed timestamps and geocoded incident locations for spatial analysis
- Removed records with missing coordinates or timestamps

### Spatial clustering

- Applied DBSCAN with haversine distance metric to identify dense incident clusters without pre-specifying cluster count
- Applied KMeans for comparison and grid-based hotspot assignment
- Mapped clusters to geographic zones for interpretability

### Temporal feature engineering

- Engineered cyclical time features (sine/cosine encoding for hour and day-of-week)
- Created rush-hour flags, weekend indicators, and seasonal markers
- Built a binary target: peak vs. non-peak incident periods

### Classification and evaluation

- Trained Random Forest and Gradient Boosting classifiers for temporal peak prediction
- Evaluated with accuracy, precision, recall, F1, and 5-fold cross-validated F1
- Gradient Boosting achieved ~80% accuracy with CV F1 of ~0.77

### Interactive dashboard

- Built a Streamlit dashboard with hotspot maps and temporal analysis views
- Integrated Plotly for interactive cluster visualization and time-series exploration

---

## Acknowledgements

- [City of Calgary Open Data Portal](https://data.calgary.ca/) for providing the traffic incident dataset
- [Socrata Open Data API](https://dev.socrata.com/) for programmatic data access

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=120&section=footer" width="100%" />
</p>
