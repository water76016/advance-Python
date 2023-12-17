# 短周期趋势预测算法

import pandas as pd  
from fbprophet import Prophet


data = pd.read_csv("timeseries_data.csv")
