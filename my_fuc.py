import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

def series_to_df(series,name_str):#series:temp_advice.groupby('药品通用名称').size()
  df = series.to_frame()
  df.reset_index(inplace=True)
  df = df.rename(columns={0 : name_str})
  df.sort_values(by='count',ascending=False,inplace=True)
  return df

def my_pca(data_df):
  data_df = data_df.copy()
  # 生成可重现的随机数据
  np.random.seed(42)
  # 获取数据维度的最小值
  min_dimension = min(data_df.shape)

  # 使用交叉验证选择最佳维度
  best_n_components = None
  best_score = -float('inf')  # 初始化最佳得分为负无穷
  for n_components in range(1, min_dimension + 1):
    pca = PCA(n_components=n_components)
    scores = cross_val_score(pca, data_df.T, cv=5)  # 交叉验证计算得分
    avg_score = np.mean(scores)
    if avg_score > best_score:
      best_score = avg_score
      best_n_components = n_components
  print("最佳维度:", best_n_components)
  # 使用最佳维度进行PCA
  pca = PCA(n_components=best_n_components)
  pca_result = pca.fit_transform(data_df.T)
  # # 输出PCA结果
  # print("PCA结果:")
  # print(pd.DataFrame(pca_result, columns=[f'PC{i + 1}' for i in range(best_n_components)]))
  # # 输出主成分的贡献率
  # print("主成分的贡献率:")
  # print(pca.explained_variance_ratio_)
  # 输出特征筛选后的基因
  selected_genes = [data_df.columns[i] for i in np.argsort(pca.components_[0])[::-1][:best_n_components]]
  return selected_genes
