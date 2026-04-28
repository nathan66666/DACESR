# import matplotlib.pyplot as plt
# import numpy as np

# # Example data
# x = np.arange(0, 1000, 1)  # Degradation values
# y_ram = np.exp(-x / 300) * 0.2 + 0.7  # RAM: Exponential decay with a constant factor
# y_dape = np.ones_like(x) * 0.7 + (np.random.rand(len(x)) - 0.5) * 0.1  # DAPE: Random noise around 0.7

# # Create plot
# plt.figure(figsize=(8, 6))

# plt.scatter(x, y_ram, label="RAM", color='blue', s=10)  # RAM points in blue
# plt.scatter(x, y_dape, label="DAPE", color='orange', s=10)  # DAPE points in orange

# # Add labels and legend
# plt.xlabel('Degradation')
# plt.ylabel('Similarity')
# plt.legend(loc="upper right")
# plt.ylim(0, 1)  # Set y-axis range from 0 to 1

# # Save the figure
# plt.savefig('/lxy/DASR/scripts/output.png')  # Save the plot as a PNG file

# # 读取一个txt文件
# import matplotlib.pyplot as plt

# # 初始化数据列表
# degrees = []
# values = []

# # 读取 ram.txt 文件
# with open('/lxy/lxy/recognize-anything/text/DAPE/a001_jaccard_avg.txt', 'r') as file:
#     for line in file:
#         # 拆分每行数据，提取度数和数值
#         degree, value = line.split()
#         degrees.append(int(degree.split('_')[1]))  # 提取数字部分
#         values.append(float(value))  # 将值转换为浮点数

# # 绘制散点图
# plt.figure(figsize=(8, 6))
# plt.scatter(degrees, values, color='blue', label='RAM')

# # 设置标题和轴标签
# plt.title('Scatter Plot of RAM Data')
# plt.xlabel('Degree')
# plt.ylabel('Value')

# # 显示图例
# plt.legend()

# plt.savefig('/lxy/DASR/scripts/output.png') 

# # 读取两个txt文件
# import matplotlib.pyplot as plt

# # 初始化数据列表
# degrees_ram = []
# values_ram = []
# degrees_dape = []
# values_dape = []

# # 读取 ram.txt 文件
# with open('/lxy/lxy/recognize-anything/text/DAPE/cls2_jaccard_avg.txt', 'r') as file:
#     for line in file:
#         degree, value = line.split()
#         degrees_ram.append(int(degree.split('_')[1]))  # 提取 deg_X 中的数字部分
#         values_ram.append(float(value))  # 将值转换为浮点数

# # 读取第二个 txt 文件（假设名为 dape.txt）
# with open('/lxy/lxy/recognize-anything/text/DAPE/cls1_jaccard_avg.txt', 'r') as file:
#     for line in file:
#         degree, value = line.split()
#         degrees_dape.append(int(degree.split('_')[1]))
#         values_dape.append(float(value))

# # 创建图表
# plt.figure(figsize=(8, 6))

# # 绘制第一个数据集 (RAM)
# plt.scatter(degrees_ram, values_ram, color='blue', label='RAM')

# # 绘制第二个数据集 (DAPE)
# plt.scatter(degrees_dape, values_dape, color='orange', label='DAPE')

# # 设置标题和轴标签
# plt.title('Scatter Plot of RAM and DAPE Data')
# plt.xlabel('Degree')
# plt.ylabel('Value')

# # 显示图例
# plt.legend()

# plt.savefig('/lxy/DASR/scripts/output.png') 


# import pandas as pd
# import matplotlib.pyplot as plt

# # 读取 plot1.txt 文件
# data = pd.read_csv('/lxy/DASR/scripts/plot2.txt', sep='\t')  # 使用制表符 ('\t') 作为分隔符

# # 提取 RAM 和 DAPE 列的数据
# degrees = data.index  # 使用行号作为 x 轴的 degree
# ram_values = data['RAM']
# dape_values = data['DAPE']

# # 创建散点图
# plt.figure(figsize=(8, 6))

# # 绘制 RAM 数据的散点图
# plt.scatter(degrees, ram_values, color='#90EE90', label='DAPE-RAM < 0')

# # 绘制 DAPE 数据的散点图
# plt.scatter(degrees, dape_values, color='#006400', label='DAPE-RAM > 0')

# # 添加标题和标签
# # plt.title('Scatter Plot of RAM and DAPE from plot1.txt')
# plt.xlabel('Degradation')
# plt.ylabel('The difference in similarity between output of DAPE and RAM')

# # 显示图例
# plt.legend()

# # 显示图表
# plt.savefig('/lxy/DASR/scripts/output2.png') 










# import pandas as pd
# import matplotlib.pyplot as plt

# plt.rcParams['font.family'] = 'DejaVu Sans'  # 设置字体家族
# plt.rcParams['font.size'] = 16               # 设置统一字体大小


# # 读取txt文件
# file_path = '/lxy/DASR/scripts/zhu1.txt'
# df = pd.read_csv(file_path, sep='\t')

# # 设置柱状图的宽度
# bar_width = 0.2
# index = range(len(df))

# # 画柱状图
# fig, ax = plt.subplots(figsize=(10, 6))

# # colors = ['#FDAE6B', '#E6550D', '#A63603', '#7F2704']

# # 添加柱状数据，调整不同类别柱状的相对位置
# ax.bar([i - 1.5 * bar_width for i in index], df['RAM'], width=bar_width, label='RAM')
# ax.bar([i - 0.5 * bar_width for i in index], df['DAPE'], width=bar_width, label='DAPE')
# ax.bar([i + 0.5 * bar_width for i in index], df['cls1'], width=bar_width, label='DAPE-FT-1')
# ax.bar([i + 1.5 * bar_width for i in index], df['cls4'], width=bar_width, label='DAPE-FT-2')

# ax.set_ylim(0.5, None)

# # 添加标签和标题
# # ax.set_xlabel('Degradation (Range)')
# ax.set_ylabel('similarity')
# # ax.set_title('Comparison of RAM, DAPE')
# ax.set_xticks(index)
# ax.set_xticklabels(df['degradation'])

# # 添加图例
# ax.legend()

# # 显示图形
# plt.tight_layout()
# plt.savefig('/lxy/DASR/scripts/output_zhu1_1_rq.png')


import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'  # 设置字体家族
plt.rcParams['font.size'] = 16               # 设置统一字体大小

# 读取txt文件
file_path = '/lxy/DASR/scripts/zhu_blur.txt'
df = pd.read_csv(file_path, sep='\t', index_col=0)

# 设置柱状图的宽度
bar_width = 0.2
index = range(len(df.columns))  # 每个退化类别的索引

# 画柱状图
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制每个模型的数据，使用不同的颜色和位置
ax.bar([i - 1.5 * bar_width for i in index], df.loc['RAM'], width=bar_width, label='RAM')
ax.bar([i - 0.5 * bar_width for i in index], df.loc['DAPE'], width=bar_width, label='DAPE')
ax.bar([i + 0.5 * bar_width for i in index], df.loc['DAPE-FT-1'], width=bar_width, label='DAPE-FT-1')
ax.bar([i + 1.5 * bar_width for i in index], df.loc['DAPE-FT-2'], width=bar_width, label='DAPE-FT-2')

# 设置 y 轴的起始点
ax.set_ylim(0.50, None)

# 添加标签和标题
# ax.set_xlabel('Degradation class')
ax.set_ylabel('Similarity')
ax.set_xticks(index)
ax.set_xticklabels(df.columns, rotation=45)  # 使用列名作为 x 轴标签，并旋转标签以避免重叠

# 添加图例
ax.legend(fontsize=16)

plt.xticks(rotation=0)  # rotation=0 表示水平显示
# 显示图形并保存
plt.tight_layout()
plt.savefig('/lxy/DASR/scripts/output_zhu_blur.png')
plt.show()
