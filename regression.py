# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:00:36 2019

@author: Shanks
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

#读入数据
df = pd.read_excel("C:\\Users\Shanks\desktop\xxxxxxxxx.xslx", )
df.head()
df.fillna(df.mean(axis=0))

#画图对数据进行描述
import matplotlib as mpl  #显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']  #配置显示中文，否则乱码
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号，如果是plt画图，则将mlp换成plt
sns.pairplot(df, x_vars=['中证500','泸深300','上证50','上证180'], y_vars='上证指数',kind="reg", size=5, aspect=0.7)
plt.show()


#抽取X，dataframe格式即可，抽取y，series格式即可
X = df.iloc[]
y = df.iloc[]
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=100)

#开始回归
from sklearn.model_selection import train_test_split #这里是引用了交叉验证
from sklearn.linear_model import LinearRegression  #线性回归
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

linreg = LinearRegression()
model=linreg.fit(X_train, y_train)
print(model)
# 训练后模型截距
print(linreg.intercept_)
# 训练后模型权重（特征个数无变化）
print(linreg.coef_)
#对照一下每个特征对应的系数
feature_cols = ['中证500','泸深300','上证50','上证180','上证指数']
B=list(zip(feature_cols,linreg.coef_))
print(B)

#模型评估
sum_mean=0
for i in range(len(y_pred)):
    sum_mean+=(y_pred[i]-y_test.values[i])**2
sum_erro=np.sqrt(sum_mean/10)  #这个10是你测试级的数量
# calculate RMSE by hand
print ("RMSE by hand:",sum_erro)
#做ROC曲线
plt.figure()
plt.plot(range(len(y_pred)),y_pred,'b',label="predict")
plt.plot(range(len(y_pred)),y_test,'r',label="test")
plt.legend(loc="upper right") #显示图中的标签
plt.xlabel("the number of sales")
plt.ylabel('value of sales')
plt.show()

