import numpy as np
import pandas as pd
from itertools import combinations

def compute_features(df):
    df['altitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
    return df

def compute_pairwise_features(df):
    pairs = []
    for (i1, row1), (i2, row2) in combinations(df.iterrows(), 2):
        dist = np.linalg.norm([
            row1['x'] - row2['x'],
            row1['y'] - row2['y'],
            row1['z'] - row2['z']
        ])
        pairs.append({
            'name1': row1['name'],
            'name2': row2['name'],
            'distance': dist,
            'altitude_diff': abs(row1['altitude'] - row2['altitude']),
            'speed_diff': abs(row1['speed'] - row2['speed'])
        })
    return pd.DataFrame(pairs)
