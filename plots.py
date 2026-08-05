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
    При наличии target_col:
      - если уникальных значений ≤ 100 → 2D scatter с категориальной окраской (hue)
      - если уникальных значений > 100 → 3D scatter (третья ось = target_col) + тепловая карта
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

    if target_col:
        n_unique = df[target_col].nunique(dropna=True)
        
        if n_unique > 100:
            # Числовая переменная - 3D scatter + тепловая карта
            if not pd.api.types.is_numeric_dtype(df[target_col]):
                raise TypeError(f"target_col '{target_col}' имеет >100 уникальных значений, "
                                f"но не является числовой.")
            
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(11, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(
                df[x_col], 
                df[y_col],
                df[target_col],
                c=df[target_col],
                cmap='viridis',
                alpha=0.7,
                s=40
            )
            
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label(target_col, fontsize=12)
            
            ax.set_xlabel(x_col, fontsize=11)
            ax.set_ylabel(y_col, fontsize=11)
            ax.set_zlabel(target_col, fontsize=11)
            ax.set_title(f'Зависимость {y_col} от {x_col} (цвет и ось Z = {target_col})', fontsize=13)
            
            # Ограничения осей
            if x_min is not None or x_max is not None:
                ax.set_xlim(left=x_min, right=x_max)
            if y_min is not None or y_max is not None:
                ax.set_ylim(bottom=y_min, top=y_max)
            
            plt.tight_layout()
            plt.show()
            return
        
        else:
            # Категориальная переменная - обычный 2D hue
            plt.figure(figsize=(9, 7))
            
            unique_vals = sorted(df[target_col].dropna().unique())
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
        plt.figure(figsize=(9, 7))
        sns.scatterplot(data=df, x=x_col, y=y_col, color='steelblue', alpha=0.7, s=60)

    # Ограничения осей (только для 2D)
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


def plot_box_by_category(df, cat_feature, num_feature, categories=None, 
                         figsize=(10, 6), showfliers=True, title=None):
    """
    Строит boxplot числового признака в разрезе категорий категориального признака.
    
    Параметры:
        df          - DataFrame
        cat_feature - название категориального признака (например, 'income')
        num_feature - название числового признака (например, 'hours-per-week')
        categories  - список категорий в нужном порядке (если None — берётся sorted unique)
        showfliers  - показывать ли выбросы
        title       - заголовок графика (если None — создаётся автоматически)
    """
    
    if categories is None:
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


def plot_regression_results(metrics, y_true, y_pred, model_name="Model"):
    """
    Визуализация результатов регрессии:
    1. Predicted vs Actual
    2. Residuals plot
    3. Distribution of residuals
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolor='k', s=40)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
    axes[0].set_xlabel('Actual', fontsize=12)
    axes[0].set_ylabel('Predicted', fontsize=12)
    axes[0].set_title(f'{model_name}\nPredicted vs Actual', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Residuals plot
    axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolor='k', s=40)
    axes[1].axhline(0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title(f'{model_name}\nResiduals vs Predicted', fontsize=13)
    axes[1].grid(True, alpha=0.3)

    # 3. Distribution of residuals
    sns.histplot(residuals, kde=True, ax=axes[2], color='steelblue')
    axes[2].axvline(0, color='r', linestyle='--', lw=2)
    axes[2].set_xlabel('Residuals', fontsize=12)
    axes[2].set_title(f'{model_name}\nDistribution of Residuals', fontsize=13)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_regression_report(metrics, model_name="Model"):
    """
    Красивый вывод метрик регрессии
    """
    print("=" * 50)
    print(f" Regression Report: {model_name}")
    print("=" * 50)
    print(f"MAE   : {metrics['MAE']:.4f}")
    print(f"MSE   : {metrics['MSE']:.4f}")
    print(f"RMSE  : {metrics['RMSE']:.4f}")
    print(f"R²    : {metrics['R2']:.4f}")
    print(f"MAPE  : {metrics['MAPE']:.2f}%")
    print(f"MedAE : {metrics['MedAE']:.4f}")
    print("=" * 50)


def plot_feature_importance(model, feature_names, top_n=None, figsize=(10, 6),
                            model_type='auto'):
    """
    Plot feature importance for various regression model types using Seaborn.

    Parameters:
    - model: Trained model (DecisionTreeRegressor, RandomForestRegressor, LinearRegression, Ridge, Lasso и т.д.)
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
        # Для регрессии coef_ обычно одномерный
        coef = model.coef_
        if coef.ndim > 1:
            # На всякий случай (если вдруг multi-output)
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
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


def visualize_decision_tree(model, feature_names, figsize=(20, 10), max_depth=None):
    """
    Visualize the decision tree structure (for DecisionTreeRegressor).

    Parameters:
    - model: Trained DecisionTreeRegressor
    - feature_names: List of feature names
    - figsize: Figure size
    - max_depth: Maximum depth to display (None for full tree)
    """
    plt.figure(figsize=figsize)
    plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        proportion=True,
        max_depth=max_depth,
        impurity=True          # показывает MSE / variance
    )
    plt.title('Decision Tree Regressor Visualization')
    plt.show()


def plot_hyperparam_search_results(
    results,
    score_key='mean_test_score',
    title='Hyperparameter Tuning Results',
    xtick_step=5,
    higher_is_better=True
):
    """
    Generic plot function for hyperparameter search results from GridSearchCV, 
    RandomizedSearchCV, BayesSearchCV и т.д.

    Args:
        results (dict or pd.DataFrame): Search results. Must contain 'params' and score_key.
        score_key (str): Key for the score column (default 'mean_test_score').
        title (str): Plot title.
        xtick_step (int): Frequency of x-axis labels.
        higher_is_better (bool): True для R², False для MAE/MSE/RMSE (когда используется neg_*)
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
    if higher_is_better:
        best_idx = df[score_key].idxmax()
    else:
        best_idx = df[score_key].idxmin()
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
                 xytext=(df.loc[best_idx, 'Set #'], best_score + (0.02 if higher_is_better else -0.02)),
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
    Работает одинаково и для классификации, и для регрессии.
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