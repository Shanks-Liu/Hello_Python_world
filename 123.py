# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:11:07 2018

@author: 游侠-Speed
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as scs
import matplotlib.pyplot as plt

stock_set = ['000598.XSHE','002258.XSHE','600855.XSHG','600649.XSHG','600827.XSHG']
noa = len(stock_set)
start_date='2012-12-31'
end_date='2016-12-30'
df = get_price(stock_set, start_date, end_date, 'daily', ['close'])
data = df['close']
#规范化后时序数据
(data/data.ix[0]*100).plot(figsize = (8,5))