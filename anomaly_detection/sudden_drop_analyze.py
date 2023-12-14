# 骤降分析

import numpy as np 
import pandas as pd 
import os 
from pyod.models.lof import LOF 

#获取训练模型 
def getTrainModel(list_data):
    model = LOF(n_neighbors = 10)
    train_data = np.array(list_data).reshape(-1, 1)
    model.fit(train_data)
    return model 
  
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