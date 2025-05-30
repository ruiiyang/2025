import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_relationships(
    data: pd.DataFrame,
    target: str = 'Y',
    features: list[str] = ['Ya', 'Yb', 'Yc']
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Correlation and mutual info calculation and build visualisation

    Returns:
        corr_df: DataFrame，includes Pearson and Spearman coefficients
        mi_df:   DataFrame，includes mutual information
    """

    # calculate correlation
    corr_results = []
    for col in features:
        pearson_r, pearson_p = stats.pearsonr(data[target], data[col])
        spearman_r, spearman_p = stats.spearmanr(data[target], data[col])
        corr_results.append({
            'Feature': col,
            'Pearson_r': f"{pearson_r:.3f} (p={pearson_p:.3f})",
            'Spearman_r': f"{spearman_r:.3f} (p={spearman_p:.3f})"
        })
    corr_df = pd.DataFrame(corr_results)

    # calculate mutual info
    mi = mutual_info_regression(data[features], data[target])
    mi_df = pd.DataFrame({'Feature': features, 'Mutual_Info': mi})

    #visualisation
    #heat map
    n = data.shape[1]
    size = max(8, n)
    plt.figure(figsize=(size, size))
    sns.heatmap(
        data.corr(),
        annot=True,
        cmap='coolwarm',
        annot_kws={"size": 10}
    )
    plt.title("Correlation Heatmap")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return corr_df, mi_df