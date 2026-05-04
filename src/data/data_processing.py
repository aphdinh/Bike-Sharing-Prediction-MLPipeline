import pandas as pd
import numpy as np
import logging
import os
from ..utils.aws_utils import download_from_s3

NUMERICAL_FEATURES = [
    'hour', 'temperature_c', 'humidity', 'wind_speed', 'visibility_10m',
    'dew_point_c', 'solar_radiation', 'rainfall_mm', 'snowfall_cm',
    'year', 'month', 'day_of_week', 'hour_sin', 'hour_cos',
    'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
    'temp_humidity_interaction', 'wind_rain_interaction', 'temp_solar_interaction'
]

BINARY_FEATURES = [
    'is_weekend', 'is_rush_hour', 'has_rain', 'has_snow',
    'is_holiday', 'is_functioning', 'is_spring', 'is_summer',
    'is_autumn', 'is_winter', 'extreme_weather'
]

CATEGORICAL_FEATURES = ['time_of_day', 'temp_feel']

EXPECTED_CATEGORICAL_COLS = [
    'time_of_day_Night', 'time_of_day_Morning', 'time_of_day_Afternoon', 'time_of_day_Evening',
    'temp_feel_Very_Cold', 'temp_feel_Cold', 'temp_feel_Mild', 'temp_feel_Warm'
]


def load_data(file_path='data/SeoulBikeData.csv'):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding='latin1')
    else:
        logging.info(f"Local file not found, attempting to download from S3...")
        if download_from_s3(f"data/{file_path}", file_path):
            df = pd.read_csv(file_path, encoding='latin1')
        else:
            raise FileNotFoundError(f"Data file {file_path} not found locally or in S3")

    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'Date': 'date',
        'Rented Bike Count': 'rented_bike_count',
        'Hour': 'hour',
        'Temperature(°C)': 'temperature_c',
        'Humidity(%)': 'humidity',
        'Wind speed (m/s)': 'wind_speed',
        'Visibility (10m)': 'visibility_10m',
        'Dew point temperature(°C)': 'dew_point_c',
        'Solar Radiation (MJ/m2)': 'solar_radiation',
        'Rainfall(mm)': 'rainfall_mm',
        'Snowfall (cm)': 'snowfall_cm',
        'Seasons': 'season',
        'Holiday': 'holiday',
        'Functioning Day': 'functioning_day'
    })
    return df


def feature_engineering(df):
    df = df.copy()

    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_name'] = df['date'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    df['time_of_day'] = pd.cut(df['hour'],
                               bins=[0, 6, 12, 18, 24],
                               labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                               include_lowest=True)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df['temp_feel'] = pd.cut(df['temperature_c'],
                             bins=[-float('inf'), 0, 10, 20, float('inf')],
                             labels=['Very_Cold', 'Cold', 'Mild', 'Warm'])

    df['temp_humidity_interaction'] = df['temperature_c'] * df['humidity'] / 100
    df['wind_rain_interaction'] = df['wind_speed'] * df['rainfall_mm']
    df['temp_solar_interaction'] = df['temperature_c'] * df['solar_radiation']

    df['has_rain'] = (df['rainfall_mm'] > 0).astype(int)
    df['has_snow'] = (df['snowfall_cm'] > 0).astype(int)
    df['is_holiday'] = (df['holiday'] == 'Holiday').astype(int)
    df['is_functioning'] = (df['functioning_day'] == 'Yes').astype(int)

    df['is_spring'] = (df['season'] == 'Spring').astype(int)
    df['is_summer'] = (df['season'] == 'Summer').astype(int)
    df['is_autumn'] = (df['season'] == 'Autumn').astype(int)
    df['is_winter'] = (df['season'] == 'Winter').astype(int)

    df['extreme_weather'] = ((df['rainfall_mm'] > 5) |
                             (df['snowfall_cm'] > 1) |
                             (df['wind_speed'] > 5)).astype(int)
    return df


def prepare_features(df):
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)
    encoded_cols = [c for c in df_encoded.columns if any(cat in c for cat in CATEGORICAL_FEATURES)]
    all_features = NUMERICAL_FEATURES + BINARY_FEATURES + encoded_cols
    X = df_encoded[all_features]
    y = df_encoded['rented_bike_count']
    return X, y, all_features


def preprocess_data(df):
    df_processed = feature_engineering(df)

    for cat in CATEGORICAL_FEATURES:
        if cat not in df_processed.columns:
            df_processed[cat] = 'Morning' if cat == 'time_of_day' else 'Mild'

    df_encoded = pd.get_dummies(df_processed, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)

    for col in EXPECTED_CATEGORICAL_COLS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    encoded_cols = [c for c in df_encoded.columns if any(cat in c for cat in CATEGORICAL_FEATURES)]
    all_features = NUMERICAL_FEATURES + BINARY_FEATURES + encoded_cols
    return df_encoded[all_features]
