# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 21:29:16 2020

@author: XVZ
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_excel(r'C:\Users\XVZ\Desktop\code\finall\month-chafen.xlsx')
df = dataset.iloc[12:,1:].values
# ensure all data is float所有数据转变为float类型
df = df.astype('float32')
# normalize features归一化
scaler = MinMaxScaler(feature_range=(0, 1)).fit(df)
scaled = scaler.transform(df)
y = scaled[:,-1]
def td(x,y,n,p):
    d=asum=bsum=csum=0.0
    for l in range(-1,2):
        xmeans=np.mean(x)
        ymeans=np.mean(y)
        for i in range(l+1,n-1):
            a=(x[i]-xmeans)*(y[i+l]-ymeans)
            b=pow((x[i]-xmeans),2)
            c=pow((y[i+l]-ymeans),2)
            asum=a+asum
            bsum=b+bsum
            csum +=c
        d=np.sqrt(bsum)
        if asum==d==0:
            r=0
        else:
            r=asum/d
        rl=abs(r)
        print(p,l,rl)
for p in range(1,scaled.shape[1]):
    td(scaled[:,p],y,y.shape[0],p)