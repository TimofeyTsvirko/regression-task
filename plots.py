import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
from matplotlib.colors import LinearSegmentedColormap
import phik


def plot_phik(data, figsize=(12, 8)):
    phik_matrix = data.phik_matrix()
    plt.figure(figsize=(10, 8))
    sns.heatmap(phik_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.show()


def plot_hist_numeric(data, feature, figsize=(10, 5), x_min=None, x_max=None, bins=None):
    filtered_data = data.copy()
    if x_min is not None:
        filtered_data = filtered_data[filtered_data[feature] >= x_min]
    if x_max is not None:
        filtered_data = filtered_data[filtered_data[feature] <= x_max]

    if bins is None:
        if filtered_data[feature].nunique() < 100:
            bins = int(filtered_data[feature].max() - filtered_data[feature].min() + 1)
        else:
            bins = 'auto'

    plt.figure(figsize=figsize)
    plt.grid(True, alpha=0.3)
    
    sns.histplot(
        filtered_data[feature], 
        kde=True, 
        bins=bins,
        kde_kws={'bw_adjust': 0.8}
    )
    
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


def plot_hist_categorical(data, feature, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    category_counts = data[feature].value_counts().sort_values(ascending=False)
    category_percent = category_counts / category_counts.sum() * 100

    sns.barplot(
        x=category_counts.values,
        y=category_counts.index,
        hue=category_counts.index,
        palette="viridis",
        legend=False,
        order=category_counts.index,
        ax=ax
    )

    offset = category_counts.max() * 0.01
    for i, (count, pct) in enumerate(zip(category_counts.values, category_percent.values)):
        ax.text(
            count + offset,
            i,
            f"{pct:.1f}%",
            va="center",
            ha="left"
        )

    ax.set_title(title if title else f"Distribution of {feature}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel(feature)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, category_counts.max() * 1.15)
    return ax


def plot_categorical_relationship(df, col1, col2):
    # Абсолютные значения
    count_crosstab = pd.crosstab(df[col1], df[col2])

    # Доли по строкам (внутри col1)
    row_prop = pd.crosstab(df[col1], df[col2], normalize='index')

    # Доли по столбцам (внутри col2)
    col_prop = pd.crosstab(df[col1], df[col2], normalize='columns')

    # Фигура с 3 подграфиками по горизонтали
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # 1. Абсолютные значения
    sns.heatmap(count_crosstab, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f'Абсолютные значения\n{col1} vs {col2}')
    axes[0].set_xlabel(col2)
    axes[0].set_ylabel(col1)

    # 2. Доли внутри col1 (по строкам)
    sns.heatmap(row_prop, annot=True, fmt=".2f", cmap="Greens", ax=axes[1])
    axes[1].set_title(f'Доли внутри {col1} (по строкам)')
    axes[1].set_xlabel(col2)
    axes[1].set_ylabel(col1)

    # 3. Доли внутри col2 (по столбцам)
    sns.heatmap(col_prop, annot=True, fmt=".2f", cmap="Oranges", ax=axes[2])
    axes[2].set_title(f'Доли внутри {col2} (по столбцам)')
    axes[2].set_xlabel(col2)
    axes[2].set_ylabel(col1)

    plt.tight_layout()
    plt.show()


def plot_numeric_relationship(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    target_col: str = None,
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None
):
    """
    Строит scatter plot зависимости между двумя числовыми переменными.
    При наличии target_col точки автоматически окрашиваются по его значениям.
    """
    # Проверка колонок
    required_cols = [x_col, y_col]
    if target_col:
        required_cols.append(target_col)
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Колонка '{col}' отсутствует в DataFrame.")

    # Проверка типов
    if not pd.api.types.is_numeric_dtype(df[x_col]):
        raise TypeError(f"{x_col} не является числовой переменной.")
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise TypeError(f"{y_col} не является числовой переменной.")

    # Построение графика
    plt.figure(figsize=(9, 7))

    if target_col:
        # Автоматическое определение уникальных значений и цветов
        unique_vals = sorted(df[target_col].dropna().unique())
        
        if len(unique_vals) > 6:  # слишком много категорий
            raise ValueError(f"target_col '{target_col}' имеет слишком много уникальных значений ({len(unique_vals)}).")
        
        # Автоматическая палитра
        palette = sns.color_palette("Set2", n_colors=len(unique_vals))
        color_dict = dict(zip(unique_vals, palette))
        
        sns.scatterplot(
            data=df, 
            x=x_col, 
            y=y_col,
            hue=target_col, 
            palette=color_dict,
            alpha=0.7,
            s=60
        )
        plt.legend(title=target_col, title_fontsize=12)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, color='steelblue', alpha=0.7, s=60)

    # Ограничения осей
    if x_min is not None or x_max is not None:
        plt.xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min, top=y_max)

    plt.title(f'Зависимость {y_col} от {x_col}', fontsize=14)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_box_by_category(df, cat_feature, num_feature, figsize=(10, 6), 
                         showfliers=True, title=None):
    """
    Строит boxplot числового признака в разрезе категорий категориального признака.
    
    Параметры:
        df          - DataFrame
        cat_feature - название категориального признака (например, 'income')
        num_feature - название числового признака (например, 'hours-per-week')
        showfliers  - показывать ли выбросы
        title       - заголовок графика (если None — создаётся автоматически)
    """
    
    categories = sorted(df[cat_feature].dropna().unique())
    
    data_groups = [df[df[cat_feature] == cat][num_feature].dropna() 
                   for cat in categories]
    
    plt.figure(figsize=figsize)
    
    plt.boxplot(
        data_groups,
        tick_labels=categories,
        showfliers=showfliers,
        patch_artist=True
    )
    
    # Автоматический заголовок
    if title is None:
        title = f"Distribution of {num_feature} by {cat_feature}"
    
    plt.title(title, fontsize=14)
    plt.xlabel(cat_feature.capitalize(), fontsize=12)
    plt.ylabel(num_feature.replace('-', ' ').title(), fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()


def plot_classification_results(metrics, model_name="Model"):
    """
    Plot classification evaluation results

    Parameters:
    -----------
    metrics : dict
        Dictionary containing all metrics (output from calculate_classification_metrics)
    model_name : str, optional
        Name of the model for display purposes
    """
    plt.figure(figsize=(15, 6))

    # Plot 1: Confusion Matrix
    if 'Confusion Matrix' in metrics:
        plt.subplot(1, 2, 1)
        sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title(f'{model_name} - Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)

    # Plot 2: ROC Curve (if available)
    if 'ROC Curve' in metrics:
        roc_data = metrics['ROC Curve']
        plt.subplot(1, 2, 2)
        plt.plot(roc_data['fpr'], roc_data['tpr'], color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {metrics["ROC AUC"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic', fontsize=14)
        plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


def print_classification_report(metrics, model_name="Model"):
    """
    Print classification evaluation report

    Parameters:
    -----------
    metrics : dict
        Dictionary containing all metrics (output from calculate_classification_metrics)
    model_name : str, optional
        Name of the model for display purposes
    """
    # Create metrics table
    metrics_df = pd.DataFrame({
        'Metric': ['ROC AUC', 'F1 Score', 'Precision', 'Recall', 'Accuracy'],
        'Value': [
            f'{metrics["ROC AUC"]:.4f}' if metrics["ROC AUC"] is not None else 'N/A',
            f'{metrics["F1 Score"]:.4f}',
            f'{metrics["Precision"]:.4f}',
            f'{metrics["Recall"]:.4f}',
            f'{metrics["Accuracy"]:.4f}'
        ]
    })

    # Classification report dataframe
    class_report_df = pd.DataFrame(metrics['Classification Report'])

    # Display results
    print("\n" + "="*60)
    print(f"{model_name.upper()} EVALUATION".center(60))
    print("="*60)

    print("\nMAIN METRICS:")
    print(metrics_df.to_string(index=False))

    print("\n\nCLASSIFICATION REPORT:")
    print(class_report_df.to_string(index=False))

    print("\n" + "="*60)


def plot_feature_importance(model, feature_names, top_n=None, figsize=(10, 6),
                            model_type='auto'):
    """
    Plot feature importance for various model types using Seaborn.

    Parameters:
    - model: Trained model (DecisionTree, RandomForest, LogisticRegression, etc.)
    - feature_names: List of feature names
    - top_n: Show only top N important features (None for all)
    - figsize: Figure size
    - model_type: 'auto' (default), 'tree', or 'linear'. If 'auto', tries to determine automatically
    """
    # Determine model type if auto
    if model_type == 'auto':
        if hasattr(model, 'feature_importances_'):
            model_type = 'tree'
        elif hasattr(model, 'coef_'):
            model_type = 'linear'
        else:
            raise ValueError(
                "Could not determine model type automatically. Please specify 'tree' or 'linear'")

    # Get feature importances based on model type
    if model_type == 'tree':
        importances = model.feature_importances_
        importance_label = "Feature Importance"
    elif model_type == 'linear':
        # For linear models, use absolute coefficients as importance
        if len(model.coef_.shape) > 1:  # multi-class
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:  # binary classification
            importances = np.abs(model.coef_[0])
        importance_label = "Absolute Coefficient"
    else:
        raise ValueError("model_type must be either 'tree' or 'linear'")

    # Create DataFrame
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Select top_n features if specified
    if top_n is not None:
        feature_imp = feature_imp.head(top_n)

    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_imp,
        hue='Feature',
        palette='viridis',
        legend=False
    )
    plt.title(f'Feature Importances ({model_type} model)')
    plt.xlabel(importance_label)
    plt.tight_layout()
    plt.show()

    return feature_imp


def visualize_decision_tree(model, feature_names, class_names=None,
                            figsize=(20, 10), max_depth=None):
    """
    Visualize the decision tree structure.

    Parameters:
    - model: Trained DecisionTree model
    - feature_names: List of feature names
    - class_names: List of class names (for classification)
    - figsize: Figure size
    - max_depth: Maximum depth to display (None for full tree)
    """
    plt.figure(figsize=figsize)
    plot_tree(model,
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              proportion=True,
              max_depth=max_depth)
    plt.title('Decision Tree Visualization')
    plt.show()


def plot_hyperparam_search_results(
    results,
    score_key='mean_test_score',
    title='Hyperparameter Tuning Results',
    xtick_step=5
):
    """
    Generic plot function for hyperparameter search results from GridSearchCV, RandomizedSearchCV,
    BayesSearchCV, or any source with similar output.

    Args:
        results (dict or pd.DataFrame): Search results. Must contain 'params' and score_key.
        score_key (str): Key for the score column (default 'mean_test_score').
        title (str): Plot title.
        xtick_step (int): Frequency of x-axis labels.
    """
    # Normalize input
    if isinstance(results, dict):
        params = results.get('params')
        scores = results.get(score_key)
        if params is None or scores is None:
            raise ValueError(
                f"'params' and '{score_key}' must exist in results dict.")
        df = pd.DataFrame(params)
        df[score_key] = scores
    elif isinstance(results, pd.DataFrame):
        if 'params' in results.columns:
            df = pd.DataFrame(results['params'].tolist())
            df[score_key] = results[score_key].values
        else:
            raise ValueError("DataFrame input must have a 'params' column.")
    else:
        raise TypeError(
            "results must be a dict (like cv_results_) or a DataFrame.")

    df = df.reset_index().rename(columns={'index': 'Set #'})

    # Best score
    best_idx = df[score_key].idxmax()
    best_score = df.loc[best_idx, score_key]

    # Plot
    plt.figure(figsize=(12, 6))
    x = df['Set #']
    y = df[score_key]
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Hyperparameter Set #")
    plt.ylabel(score_key)
    plt.grid(True)

    # Clean x-ticks
    plt.xticks(ticks=x[::xtick_step])

    # Highlight best
    plt.plot(df.loc[best_idx, 'Set #'], best_score,
             'ro', label=f'Best: {best_score:.4f}')
    plt.annotate(f'Best\n{best_score:.4f}',
                 xy=(df.loc[best_idx, 'Set #'], best_score),
                 xytext=(df.loc[best_idx, 'Set #'], best_score + 0.02),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 ha='center')

    plt.legend()
    plt.tight_layout()
    plt.show()

    return df


def compare_metrics_heatmap(df1, df2, df1_name='DF1', df2_name='DF2',
                            figsize=(8, 4), annot_fontsize=10,
                            title='Comparison of ML Metrics'):
    """
    Compare two DataFrames of ML metrics and plot a heatmap of their differences.

    Parameters:
    - df1, df2: DataFrames containing metrics for ML algorithms (algorithms as index, metrics as columns)
    - df1_name, df2_name: Names to display for each DataFrame in the comparison
    - figsize: Size of the output figure
    - annot_fontsize: Font size for annotations in heatmap
    - title: Title for the plot

    Returns:
    - A matplotlib Figure object
    - The delta DataFrame showing the differences
    """

    # Calculate delta (difference) between DataFrames
    delta = df2 - df1

    # Create a custom red-white-green colormap
    colors = ["#ff2700", "#ffffff", "#00b975"]  # Red -> White -> Green
    cmap = LinearSegmentedColormap.from_list("rwg", colors)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        delta,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        center=0,
        linewidths=.5,
        ax=ax,
        annot_kws={"size": annot_fontsize},
        cbar_kws={'label': f'Difference ({df2_name} - {df1_name})'}
    )

    # Customize plot
    ax.set_title(title, pad=20, fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    return fig, delta
