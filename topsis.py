# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 21:19:55 2018

@author: 游侠-Speed
"""


import pandas as pd
import numpy as np


def topsis(data, index):

    # 最优最劣方案
    Z_positive = data.max()
    Z_negative = data.min()

    # 距离
    D_positive = np.sqrt(((data - Z_positive) ** 2).sum(axis=1))
    D_negative = np.sqrt(((data - Z_negative) ** 2).sum(axis=1))
    
    # 贴近程度
    C = D_negative / (D_negative + D_positive)
    out = pd.DataFrame({'最终得分': C}, index=index)
    #out['排序'] = out.rank(ascending=False)['最终得分']

    print(out)
    return out

file = r'C:\Users\Shanks\Desktop\所有实证返回结果.xlsx'
df = pd.read_excel(file, sheet_name=8, index_col=0)
index = df.index
data = df.values
out = topsis(data, index)