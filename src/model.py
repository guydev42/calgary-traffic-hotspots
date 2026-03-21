"""
Model module for Calgary Traffic Incident Hotspot Analysis.

Implements spatial clustering (DBSCAN, KMeans) to identify traffic incident
hotspots and classification models (RandomForest, GradientBoosting) to
predict high-incident hours and locations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# Spatial Clustering
# ---------------------------------------------------------------------------

class SpatialClusterAnalyzer:
    """
    Performs spatial clustering on traffic incident locations using
    DBSCAN and KMeans algorithms.
    """

    def __init__(self):
        """Initialize the spatial cluster analyzer."""
        self.dbscan_model = None
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.dbscan_labels_ = None
        self.kmeans_labels_ = None

    def fit_dbscan(
        self,
        coordinates: np.ndarray,
        eps: float = 0.005,
        min_samples: int = 10,
    ) -> np.ndarray:
        """
        Fit a DBSCAN clustering model on geographic coordinates.

        Parameters
        ----------
        coordinates : np.ndarray
            Array of shape (n_samples, 2) with [latitude, longitude].
        eps : float, optional
            Maximum distance between two points to be considered neighbors.
            For lat/lon in degrees, 0.005 is roughly 500 meters.
        min_samples : int, optional
            Minimum number of points to form a dense region.

        Returns
        -------
        np.ndarray
            Cluster labels for each sample (-1 indicates noise).
        """
        logger.info(
            "Fitting DBSCAN with eps=%.4f, min_samples=%d on %d points.",
            eps,
            min_samples,
            len(coordinates),
        )
        self.dbscan_model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="haversine",
            algorithm="ball_tree",
        )

        # Convert degrees to radians for haversine metric
        coords_rad = np.radians(coordinates)
        self.dbscan_labels_ = self.dbscan_model.fit_predict(coords_rad)

        n_clusters = len(set(self.dbscan_labels_)) - (
            1 if -1 in self.dbscan_labels_ else 0
        )
        n_noise = (self.dbscan_labels_ == -1).sum()
        logger.info(
            "DBSCAN found %d clusters and %d noise points.", n_clusters, n_noise
        )
        return self.dbscan_labels_

    def fit_kmeans(
        self,
        coordinates: np.ndarray,
        n_clusters: int = 8,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Fit a KMeans clustering model on geographic coordinates.

        Parameters
        ----------
        coordinates : np.ndarray
            Array of shape (n_samples, 2) with [latitude, longitude].
        n_clusters : int, optional
            Number of clusters to form.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Cluster labels for each sample.
        """
        logger.info(
            "Fitting KMeans with n_clusters=%d on %d points.",
            n_clusters,
            len(coordinates),
        )
        scaled_coords = self.scaler.fit_transform(coordinates)
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300,
        )
        self.kmeans_labels_ = self.kmeans_model.fit_predict(scaled_coords)

        logger.info("KMeans clustering complete with %d clusters.", n_clusters)
        return self.kmeans_labels_

    def get_cluster_centers(self, method: str = "kmeans") -> Optional[np.ndarray]:
        """
        Retrieve cluster center coordinates.

        Parameters
        ----------
        method : str
            Clustering method ("kmeans" or "dbscan").

        Returns
        -------
        np.ndarray or None
            Cluster center coordinates in original lat/lon space,
            or None if the model has not been fitted.
        """
        if method == "kmeans" and self.kmeans_model is not None:
            centers_scaled = self.kmeans_model.cluster_centers_
            centers = self.scaler.inverse_transform(centers_scaled)
            return centers
        elif method == "dbscan" and self.dbscan_labels_ is not None:
            logger.info(
                "DBSCAN does not have explicit centers; "
                "computing median of each cluster."
            )
            return None
        return None

    def get_cluster_summary(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        label_column: str = "cluster",
    ) -> pd.DataFrame:
        """
        Generate a summary of each cluster.

        Parameters
        ----------
        df : pd.DataFrame
            Original dataframe (must be same length as labels).
        labels : np.ndarray
            Cluster labels.
        label_column : str
            Name for the cluster label column.

        Returns
        -------
        pd.DataFrame
            Summary with count, center lat/lon per cluster.
        """
        temp = df.copy()
        temp[label_column] = labels

        summary = (
            temp.groupby(label_column)
            .agg(
                incident_count=("latitude", "size"),
                center_lat=("latitude", "median"),
                center_lon=("longitude", "median"),
            )
            .reset_index()
            .sort_values("incident_count", ascending=False)
        )
        return summary


# ---------------------------------------------------------------------------
# Classification Models
# ---------------------------------------------------------------------------

class IncidentClassifier:
    """
    Trains and evaluates classification models to predict whether a
    given hour-location combination will have high incident frequency.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the incident classifier.

        Parameters
        ----------
        random_state : int
            Random seed for reproducibility.
        """
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.feature_names: List[str] = []
        self.best_model_name: Optional[str] = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _build_models(self) -> Dict[str, Any]:
        """
        Create a dictionary of classification models.

        Returns
        -------
        dict
            Mapping of model names to sklearn estimator instances.
        """
        return {
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state,
            ),
        }

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all classifiers and evaluate them on a hold-out test set.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Binary target labels.
        test_size : float, optional
            Fraction of data reserved for testing (default 0.2).

        Returns
        -------
        dict
            Nested dictionary of model_name -> metric_name -> value.
        """
        self.feature_names = list(X.columns)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        logger.info(
            "Training set: %d samples | Test set: %d samples",
            len(self.X_train),
            len(self.X_test),
        )

        model_definitions = self._build_models()
        best_f1 = -1.0

        for name, model in model_definitions.items():
            logger.info("Training %s ...", name)
            try:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

                metrics = {
                    "accuracy": round(accuracy_score(self.y_test, y_pred), 4),
                    "precision": round(
                        precision_score(self.y_test, y_pred, zero_division=0), 4
                    ),
                    "recall": round(
                        recall_score(self.y_test, y_pred, zero_division=0), 4
                    ),
                    "f1_score": round(
                        f1_score(self.y_test, y_pred, zero_division=0), 4
                    ),
                }

                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X, y, cv=5, scoring="f1", n_jobs=-1
                )
                metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
                metrics["cv_f1_std"] = round(cv_scores.std(), 4)

                self.models[name] = model
                self.results[name] = metrics

                logger.info("%s -> %s", name, metrics)

                if metrics["f1_score"] > best_f1:
                    best_f1 = metrics["f1_score"]
                    self.best_model_name = name

            except Exception as exc:
                logger.error("Error training %s: %s", name, exc)
                self.results[name] = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "cv_f1_mean": 0.0,
                    "cv_f1_std": 0.0,
                }

        logger.info("Best model: %s (F1=%.4f)", self.best_model_name, best_f1)
        return self.results

    def get_feature_importance(
        self, model_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Extract feature importances from a trained model.

        Parameters
        ----------
        model_name : str, optional
            Name of the model. Defaults to the best performing model.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with feature names and importance scores,
            sorted descending by importance.
        """
        name = model_name or self.best_model_name
        if name is None or name not in self.models:
            logger.warning("Model '%s' not found.", name)
            return None

        model = self.models[name]
        if not hasattr(model, "feature_importances_"):
            logger.warning("Model '%s' does not expose feature importances.", name)
            return None

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        return importance_df

    def get_confusion_matrix(
        self, model_name: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Compute confusion matrix for a trained model on the test set.

        Parameters
        ----------
        model_name : str, optional
            Name of the model. Defaults to the best model.

        Returns
        -------
        np.ndarray or None
            Confusion matrix of shape (2, 2).
        """
        name = model_name or self.best_model_name
        if name is None or name not in self.models:
            return None

        y_pred = self.models[name].predict(self.X_test)
        return confusion_matrix(self.y_test, y_pred)

    def get_classification_report(
        self, model_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate a classification report for a trained model.

        Parameters
        ----------
        model_name : str, optional
            Name of the model. Defaults to the best model.

        Returns
        -------
        str or None
            Text classification report.
        """
        name = model_name or self.best_model_name
        if name is None or name not in self.models:
            return None

        y_pred = self.models[name].predict(self.X_test)
        return classification_report(
            self.y_test,
            y_pred,
            target_names=["Low Incident", "High Incident"],
        )

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Return evaluation results for all models as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Rows are models, columns are metric names.
        """
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results).T.reset_index().rename(
            columns={"index": "model"}
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, filename: str) -> Path:
    """
    Save a model object to the models/ directory using joblib.

    Parameters
    ----------
    model : Any
        Trained model or pipeline to save.
    filename : str
        Name of the file (e.g., "random_forest.joblib").

    Returns
    -------
    Path
        Full path to the saved file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / filename
    joblib.dump(model, filepath)
    logger.info("Model saved to %s", filepath)
    return filepath


def load_model(filename: str) -> Any:
    """
    Load a model object from the models/ directory.

    Parameters
    ----------
    filename : str
        Name of the file to load.

    Returns
    -------
    Any
        The loaded model object.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    model = joblib.load(filepath)
    logger.info("Model loaded from %s", filepath)
    return model


def save_all_artifacts(
    cluster_analyzer: SpatialClusterAnalyzer,
    classifier: IncidentClassifier,
) -> None:
    """
    Save all trained model artifacts to disk.

    Parameters
    ----------
    cluster_analyzer : SpatialClusterAnalyzer
        Fitted clustering analyzer.
    classifier : IncidentClassifier
        Trained incident classifier.
    """
    save_model(cluster_analyzer, "cluster_analyzer.joblib")
    save_model(classifier, "incident_classifier.joblib")
    logger.info("All artifacts saved successfully.")
