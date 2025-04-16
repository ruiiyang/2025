import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(feature_names, importances, top_n=10):
    """Plot feature importance with annotations."""
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance of Yc')

    # Annotate each bar with its Importance value
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.01 * max(importance_df['Importance']),  # Offset slightly from the bar
                p.get_y() + p.get_height() / 2.,  # Center vertically
                f'{width:.4f}',  # Format to 4 decimal places
                ha='left', va='center')

    return plt