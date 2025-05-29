from analysis import analyze_relationships
from data_process import load_data,clean_data
import matplotlib.pyplot as plt

# 加载数据
data = load_data("../../data/P1data5117.csv")
data = clean_data(data)

# 执行分析
corr_df, mi_df = analyze_relationships(data)
print("=== 相关性分析 ===")
print(corr_df)
print("\n=== 互信息分析 ===")
print(mi_df.sort_values('Mutual_Info', ascending=False))
plt.show()  # 显示可视化图形