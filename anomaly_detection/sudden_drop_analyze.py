# 骤降分析

from sklearn.datasets import make_blobs  
from sklearn.neighbors import LocalOutlierFactor  
  
# 生成样本数据  
X, _ = make_blobs(n_samples=100, centers=2, random_state=42)  
  
# 创建 LOF 模型  
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)  
  
# 对数据进行拟合  
lof.fit(X)  
  
# 预测异常值  
y_pred = lof.predict(X)  
  
# 打印预测结果  
print(y_pred)