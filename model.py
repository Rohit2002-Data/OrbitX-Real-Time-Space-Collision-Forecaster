import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def generate_synthetic_conjunction_data(pair_df):
    np.random.seed(42)
    max_dist = pair_df['distance'].max()
    pair_df = pair_df.copy()
    pair_df['risk_label'] = (
        1 - (pair_df['distance'] / max_dist)
    ) + np.random.normal(0, 0.05, size=len(pair_df))
    pair_df['risk_label'] = np.clip(pair_df['risk_label'], 0, 1)
    return pair_df

def train_model(pair_df):
    features = ['distance', 'altitude_diff', 'speed_diff']
    X = pair_df[features]
    y = pair_df['risk_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics = {
        'r2_score': r2,
        'rmse': rmse,
        'feature_importances': dict(zip(features, model.feature_importances_))
    }
    return model, metrics

def predict_risk(pair_df, model):
    features = ['distance', 'altitude_diff', 'speed_diff']
    pair_df['risk_score'] = model.predict(pair_df[features])
    return pair_df
