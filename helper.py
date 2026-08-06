import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error
)
import numpy as np
import pandas as pd
from sklearn.base import clone
import plots as p
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, KFold
from sklearn.model_selection import train_test_split


def divide_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def train_val_test_split(X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=None, stratify=None):
    if not (train_size + val_size + test_size) == 1.0:
        raise ValueError("The sum of train, val, and test sizes must equal 1.0")
        
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=stratify
    )
    
    remaining_size = train_size + val_size
    relative_val_size = val_size / remaining_size
    
    stratify_tmp = y_tmp if stratify is not None else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, 
        test_size=relative_val_size, 
        random_state=random_state, 
        stratify=stratify_tmp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_regression(y_test, y_pred, model_name="Model", enable_plot=True):
    """
    Evaluate regression performance with comprehensive metrics and visualizations

    Parameters:
    -----------
    y_test : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str, optional
        Name of the model for display purposes
    enable_plot : bool, optional
        Whether to display plots and detailed reports

    Returns:
    --------
    dict: Dictionary containing all calculated metrics
    """
    # Calculate all metrics
    metrics = calculate_regression_metrics(y_test, y_pred)

    if enable_plot:
        # Generate plots
        p.plot_regression_results(metrics, y_test, y_pred, model_name)

        # Print detailed report
        p.print_regression_report(metrics, model_name)

    # Return metrics dictionary (excluding plot data for cleaner output)
    return {k: v for k, v in metrics.items() if k not in ['Residuals', 'Prediction Error']}


def calculate_regression_metrics(y_test, y_pred):
    """
    Calculate regression performance metrics

    Parameters:
    -----------
    y_test : array-like
        True target values
    y_pred : array-like
        Predicted target values

    Returns:
    --------
    dict: Dictionary containing all calculated metrics
    """
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    residuals = y_test - y_pred

    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / np.clip(np.abs(y_test), 1e-8, None))) * 100,
        'MedAE': median_absolute_error(y_test, y_pred),
        'Residuals': residuals,
        'Prediction Error': {
            'y_true': y_test,
            'y_pred': y_pred
        }
    }

    return metrics


def train_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, seed=None):
    # Set random seed if provided and model has the parameter
    if seed is not None:
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
        if hasattr(model, 'seed'):
            model.set_params(seed=seed)

    # Train the model
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)

    # Evaluate
    metrics = evaluate_regression(
        y_test=y_test,
        y_pred=y_pred,
        model_name=model_name,
        enable_plot=False
    )

    return metrics


def train_evaluate_model_cv(model, model_name, X, y,
                            preprocessor=None, cv=5, seed=None):
    """
    Train and evaluate a regression model using cross-validation and optional preprocessing.

    Args:
        model: The model to train and evaluate
        model_name: Name of the model for reporting
        X: Features
        y: Target
        preprocessor: Preprocessing pipeline (e.g., StandardScaler, OneHotEncoder)
        cv: Number of cross-validation folds
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing evaluation metrics
    """
    # Set random seed if provided and model has the parameter
    if seed is not None:
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
        if hasattr(model, 'seed'):
            model.set_params(seed=seed)

    # Create or extend pipeline with preprocessor and model
    if isinstance(preprocessor, Pipeline):
        # If preprocessor is already a pipeline, append the model to it
        preprocessor.steps.append(('model', model))
        pipeline = preprocessor
    elif preprocessor is not None:
        # Create new pipeline with preprocessor and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
    else:
        # No preprocessor, just use the model
        pipeline = model

    # Scoring metrics for regression cross-validation
    scoring = {
        'mae': 'neg_mean_absolute_error',
        'mse': 'neg_mean_squared_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2',
        'mape': 'neg_mean_absolute_percentage_error'
    }

    # Perform cross-validation
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    metrics = {
        'MAE': -cv_results['test_mae'].mean(),
        'MSE': -cv_results['test_mse'].mean(),
        'RMSE': -cv_results['test_rmse'].mean(),
        'R2': cv_results['test_r2'].mean(),
        'MAPE': -cv_results['test_mape'].mean() * 100,  # в процентах
    }

    # Можно добавить визуализацию, если нужно
    # p.plot_regression_results(metrics, model_name)

    return metrics


def train_evaluate_models_cv(models: list, X, y, preprocessor=None, cv=5, seed=None):
    # Dictionary to store all metrics
    all_metrics = {}

    for model_name, model in models:
        # Работаем с копией модели, чтобы не изменять исходные модели
        current_model = clone(model)
        current_preprocessor = clone(preprocessor) if preprocessor is not None else None

        # Store metrics
        all_metrics[model_name] = train_evaluate_model_cv(
            current_model, model_name, X, y, current_preprocessor, cv, seed)

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')

    # Plot heatmap
    p.plot_metrics_heatmap(metrics_df)

    return metrics_df


def train_evaluate_models(models: list, X_train, y_train, X_test, y_test, seed=None):
    """
    Train and evaluate multiple regression models, then display a heatmap of the metrics.

    Parameters:
    -----------
    models : list
        List of tuples containing (model_name, model_instance)
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        DataFrame containing all evaluation metrics for all models
    """

    # Dictionary to store all metrics
    all_metrics = {}

    for model_name, model in models:
        # Работаем с копией модели
        current_model = clone(model)

        # Store metrics
        all_metrics[model_name] = train_evaluate_model(
            current_model, model_name, X_train, y_train, X_test, y_test, seed)

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')

    # Plot heatmap
    p.plot_metrics_heatmap(metrics_df)

    return metrics_df


def winsorize_outliers(df, column_name, lower_bound=None, upper_bound=None):
    df = df.copy()

    if lower_bound is not None:
        df.loc[df[column_name] < lower_bound, column_name] = lower_bound
    if upper_bound is not None:
        df.loc[df[column_name] > upper_bound, column_name] = upper_bound

    return df


def train_evaluate_models_cv_log(models, X, y, preprocessor=None, cv=5, seed=None):
    """
    Обучает модели на log(y), метрики считает на исходной шкале цены.
    """

    y_log = np.log1p(y)
    
    if isinstance(cv, int):
        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    else:
        kf = cv

    all_metrics = {}

    for model_name, model in models:
        maes, mses, rmses, r2s, mapes = [], [], [], [], []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train_log = y_log.iloc[train_idx]
            y_val = y.iloc[val_idx]
            
            current_model = clone(model)
            
            if preprocessor is not None:
                pipe = clone(preprocessor)
                pipe.steps.append(('model', current_model))
            else:
                pipe = current_model
                
            pipe.fit(X_train, y_train_log)
            y_pred = np.expm1(pipe.predict(X_val))
            
            maes.append(mean_absolute_error(y_val, y_pred))
            mses.append(mean_squared_error(y_val, y_pred))
            rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            r2s.append(r2_score(y_val, y_pred))
            mapes.append(np.mean(np.abs((y_val - y_pred) / np.clip(y_val, 1e-8, None))) * 100)
        
        all_metrics[model_name] = {
            'MAE': np.mean(maes),
            'MSE': np.mean(mses),
            'RMSE': np.mean(rmses),
            'R2': np.mean(r2s),
            'MAPE': np.mean(mapes)
        }

    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
    return metrics_df