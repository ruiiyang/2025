import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_relationships(data, target='Y', features=['Ya', 'Yb', 'Yc']):
    """执行相关性+互信息分析并生成可视化"""
    # 1. 计算相关系数
    corr_results = []
    for col in features:
        pearson_r, pearson_p = stats.pearsonr(data[target], data[col])
        spearman_r, spearman_p = stats.spearmanr(data[target], data[col])
        corr_results.append({
            'Feature': col,
            'Pearson_r': f"{pearson_r:.3f} (p={pearson_p:.3f})",
            'Spearman_r': f"{spearman_r:.3f} (p={spearman_p:.3f})"
        })
    
    # 2. 计算互信息
    mi = mutual_info_regression(data[features], data[target])
    mi_results = pd.DataFrame({'Feature': features, 'Mutual_Info': mi})
    
    # 3. 可视化
    plt.figure(figsize=(12, 4))
    # 3.1 散点图矩阵
    sns.pairplot(data, x_vars=features, y_vars=target, kind='reg')
    plt.suptitle("Scatter Plot Matrix", y=1.02)
    # 3.2 热力图
    plt.figure()
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    
    return pd.DataFrame(corr_results), mi_results