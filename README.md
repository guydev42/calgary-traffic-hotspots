# Calgary traffic incident hotspot analyzer

## Problem statement

Traffic incidents in Calgary cause delays, economic losses, and safety hazards. Understanding where and when incidents cluster is essential for city planners and emergency services. This project applies spatial clustering and temporal classification to 60,000+ traffic incidents to identify high-risk hotspots and predict peak-incident periods.

## Approach

- Fetched real-time traffic incident data from Calgary Open Data (dataset `35ra-9556`)
- Applied DBSCAN with haversine distance and KMeans for spatial clustering
- Engineered cyclical time features, rush-hour flags, and weekend indicators
- Trained Random Forest and Gradient Boosting classifiers for temporal peak prediction
- Evaluated with accuracy, precision, recall, F1, and 5-fold cross-validated F1

## Key results

| Metric | Value |
|--------|-------|
| Best model | Gradient Boosting |
| Accuracy | ~0.80 |
| CV F1 (5-fold) | ~0.77 |

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
project_03_traffic_incident_hotspots/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── notebooks/
│   └── 01_eda.ipynb
└── src/
    ├── __init__.py
    ├── data_loader.py
    └── model.py
```
