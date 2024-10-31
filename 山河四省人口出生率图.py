from pyecharts.charts import Map
from pyecharts import options as opts
import matplotlib.pyplot as plt
import pandas as pd

# 设置字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
data = pd.read_excel('山河四省.xlsx')
years = list(range(2003, 2023))
x = ['人口出生率（‰）']
birth_rates_region1 = data[x].iloc[0:20]
birth_rates_region2 = data[x].iloc[20:40]
birth_rates_region3 = data[x].iloc[40:60]
birth_rates_region4 = data[x].iloc[60:80]

# 绘制折线图
plt.figure(figsize=(12, 7))
plt.plot(years, birth_rates_region1, marker='o', label='河北省', color='blue', linestyle='-', linewidth=3, markersize=5)
plt.plot(years, birth_rates_region2, marker='o', label='山西省', color='orange', linestyle='-', linewidth=3, markersize=5)
plt.plot(years, birth_rates_region3, marker='o', label='山东省', color='green', linestyle='-', linewidth=3, markersize=5)
plt.plot(years, birth_rates_region4, marker='o', label='河南省', color='red', linestyle='-', linewidth=3, markersize=5)

# 添加图表标题和标签
# plt.title('山河四省近20年的出生率趋势', fontsize=16, fontweight='bold')
plt.xlabel('年份', fontsize=14)
plt.ylabel('出生率', fontsize=14)
plt.xticks(years, fontsize=12)  # 显示所有年份，调整字体大小
plt.yticks(fontsize=12)  # y轴字体大小
plt.legend(fontsize=12, loc='upper left')  # 显示图例
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.6)  # 自定义网格线
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.9))  # 设置y轴的刻度间隔为0.5

# 显示图形
plt.tight_layout()  # 自动调整布局
plt.show()

