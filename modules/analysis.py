import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

def analyze_predictions(history_df):
    results = {
        'crop_distribution': get_crop_distribution(history_df),
        'parameter_analysis': analyze_parameters(history_df),
        'anomalies': detect_anomalies(history_df),
        'clusters': perform_clustering(history_df)
    }
    return results

def get_crop_distribution(df):
    return df['crop'].value_counts()

def analyze_parameters(df):
    params_df = pd.DataFrame([x['parameters'] for x in df])
    return params_df.describe()

def detect_anomalies(df):
    scaler = StandardScaler()
    params_df = pd.DataFrame([x['parameters'] for x in df])
    scaled_params = scaler.fit_transform(params_df)
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    return iso_forest.fit_predict(scaled_params)

def perform_clustering(df):
    scaler = StandardScaler()
    params_df = pd.DataFrame([x['parameters'] for x in df])
    scaled_params = scaler.fit_transform(params_df)
    kmeans = KMeans(n_clusters=min(3, len(params_df)), random_state=42)
    return kmeans.fit_predict(scaled_params)