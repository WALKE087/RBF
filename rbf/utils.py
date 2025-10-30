import numpy as np
import pandas as pd

def normalize_data(X, Y):
    """Normalización Min-Max (0-1)"""
    X_num = X.select_dtypes(include=[float, int]).to_numpy() if isinstance(X, pd.DataFrame) else X
    data_min = np.min(X_num, axis=0)
    data_max = np.max(X_num, axis=0)
    X_normalized = np.zeros_like(X_num, dtype=float)
    for i in range(X_num.shape[1]):
        min_val = data_min[i]
        max_val = data_max[i]
        if max_val - min_val != 0:
            X_normalized[:, i] = (X_num[:, i] - min_val) / (max_val - min_val)
        else:
            X_normalized[:, i] = 0.0
    y_min = np.min(Y)
    y_max = np.max(Y)
    if y_max - y_min != 0:
        Y_normalized = (Y - y_min) / (y_max - y_min)
    else:
        Y_normalized = np.zeros_like(Y, dtype=float)
    return X_normalized, Y_normalized, data_min, data_max, y_min, y_max

def standardize_data(X, Y):
    """Estandarización Z-score (media=0, std=1)"""
    X_num = X.select_dtypes(include=[float, int]).to_numpy() if isinstance(X, pd.DataFrame) else X
    data_mean = np.mean(X_num, axis=0)
    data_std = np.std(X_num, axis=0)
    X_standardized = np.zeros_like(X_num, dtype=float)
    for i in range(X_num.shape[1]):
        std_val = data_std[i]
        if std_val != 0:
            X_standardized[:, i] = (X_num[:, i] - data_mean[i]) / std_val
        else:
            X_standardized[:, i] = 0.0
    
    y_mean = np.mean(Y)
    y_std = np.std(Y)
    if y_std != 0:
        Y_standardized = (Y - y_mean) / y_std
    else:
        Y_standardized = np.zeros_like(Y, dtype=float)
    return X_standardized, Y_standardized, data_mean, data_std, y_mean, y_std

def handle_missing_values(df, strategy='mean', threshold=0.5):
    """
    Maneja valores faltantes en el DataFrame
    
    Args:
        df: DataFrame con posibles valores faltantes
        strategy: 'mean', 'median', 'mode', 'drop_rows', 'drop_cols'
        threshold: porcentaje máximo de valores faltantes permitidos (para drop_cols)
    
    Returns:
        DataFrame procesado, estadísticas de valores faltantes
    """
    missing_stats = {
        'total_missing': int(df.isnull().sum().sum()),
        'missing_by_column': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    # Si no hay valores faltantes, devolver el DataFrame original
    if missing_stats['total_missing'] == 0:
        return df.copy(), missing_stats
    
    if strategy == 'drop_rows':
        df_clean = df.dropna()
        missing_stats['rows_dropped'] = len(df) - len(df_clean)
    elif strategy == 'drop_cols':
        cols_to_drop = df.columns[df.isnull().mean() > threshold]
        df_clean = df.drop(columns=cols_to_drop)
        missing_stats['columns_dropped'] = list(cols_to_drop)
    elif strategy in ['mean', 'median', 'mode']:
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    if strategy == 'mean':
                        fill_value = df_clean[col].mean()
                    elif strategy == 'median':
                        fill_value = df_clean[col].median()
                    else:  # mode
                        fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
                    df_clean[col].fillna(fill_value, inplace=True)
                else:
                    # Para columnas no numéricas, usar moda
                    fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col].fillna(fill_value, inplace=True)
    else:
        df_clean = df.copy()
    
    return df_clean, missing_stats

def get_basic_statistics(df):
    """
    Genera estadísticas básicas del DataFrame
    
    Returns:
        dict con estadísticas descriptivas
    """
    stats = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        stats['numeric_stats'][col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median()),
            'q25': float(df[col].quantile(0.25)),
            'q75': float(df[col].quantile(0.75))
        }
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    stats['categorical_info'] = {}
    for col in categorical_cols:
        stats['categorical_info'][col] = {
            'unique_values': int(df[col].nunique()),
            'top_values': df[col].value_counts().head(5).to_dict()
        }
    
    return stats

def one_hot_encode_X(X_df):
    X_processed = pd.get_dummies(X_df, drop_first=False)
    return X_processed, list(X_processed.columns)

def label_encode_Y(y_series):
    if not pd.api.types.is_numeric_dtype(y_series):
        y_cat = y_series.astype('category')
        target_mapping = dict(enumerate(y_cat.cat.categories))
        Y = y_cat.cat.codes.to_numpy(dtype=float)
    else:
        target_mapping = None
        Y = y_series.to_numpy(dtype=float)
    return Y, target_mapping

def align_features_to_model(X_df, model_feature_names):
    return X_df.reindex(columns=model_feature_names, fill_value=0.0)

