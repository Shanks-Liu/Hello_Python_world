# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 10:09:11 2018

@author: 游侠-Speed
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

class RSR(object):
    
    def __init__(self, data, *benefit_index, w=1, full_rank=True, threshold=[4, 6]):
        '''传入的数据为pd.DataFrame类型'''
        self.data = data
        self.benefit_index = benefit_index
        self.w = w
        self.full_rank = full_rank
        self.threshold = threshold
        self.result = pd.DataFrame()
        self.shape = self.data.shape
        self.ascending = [0] * self.data.shape[1]
        self.__Initialize()
        
    def __check_w(self):
        '''检查权值w'''
        if self.w == 1:
            self.w = np.array([self.w] * self.shape[1]) / self.shape[1]
        else: 
            if len(self.w) != self.shape[1]:
                self.w = np.array(list(self.w) + (self.shape[1] - len(self.w)) * [1])
            else:
                self.w = np.array(self.w)
            self.w = self.w / (sum(self.w)*self.shape[1])
            
    # 步骤1：指标同向化、排序编制
    def __isbenefit(self):
        '''效益性指标反转'''
        for index in self.benefit_index:
            self.ascending[index] = 1
    
    def __make_up(self):
        '''对原始数据排序编秩'''
        if self.full_rank:
            for i, X in enumerate(self.data.columns):
                self.result['X' + str(i + 1) + ':' + X] = self.data.iloc[:, i]
                self.result['R' + str(i + 1) + ':' + X] = self.data.iloc[:, i].rank(ascending=self.ascending[i])
        else:
            for i, X in enumerate(self.data.columns):
                self.result['X' + str(i + 1) + ':' + X] = self.data.iloc[:, i]
                self.result['R' + str(i + 1) + ':' + X] = 1 + (self.shape[0] - 1)* (self.data.iloc[:, i].max() - self.data.iloc[:, i]) / (self.data.iloc[:, i].max() - self.data.iloc[:, i].min())
                                                        
    # 步骤2：计算秩和比
    def __calculate_the_rank_sum_ratio(self):
        '''计算秩和比'''
        self.result['RSR'] = (self.result.iloc[:, 1::2] * self.w).sum(axis=1) / (self.shape[0])
        self.result['RSR_Rank'] = self.result['RSR'].rank(ascending=False)
        
        
    # 步骤3：确定RSR分布
    def __distribution_rsr_table(self):
        '''绘制RSR分布表'''
        self.distribution = pd.DataFrame(
                {'RSR': self.result['RSR'].value_counts().index.sort_values(),
                 'f': self.result['RSR'].value_counts().sort_index(),
                 '∑ f': self.result['RSR'].value_counts().sort_index().cumsum(),
                 r'\bar{R}': self.result['RSR'].sort_values().rank().value_counts().cumsum().sort_index().index,
                 r'-\bar{R}/n*100%': self.result['RSR'].sort_values().rank().value_counts().cumsum().sort_index().index /self.shape[0]},
                 columns=['RSR', 'f', '∑ f', r'\bar{R}', r'-\bar{R}/n*100%'])
        
    # 步骤4：计算回归方程并进行回归分析
    def __regression(self):
        self.r0 = np.polyfit(self.distribution['Probit'], self.distribution['RSR'], deg=1)
        est = sm.OLS(self.distribution['Probit'], self.distribution['RSR'])
        print(est.fit().summary())
        print('\n回归直线方程为：y = %f Probit + %f' % (self.r0[0], self.r0[1]))
        
    # 步骤5：代入回归方程
    def __fit_probit(self):
        # 排序
        Probit_test = pd.DataFrame({'RSR': self.distribution['RSR'], 'Probit': self.distribution['Probit']})
        self.result = pd.merge(self.result, Probit_test, on='RSR', right_index=True)
        self.result['RSR回归'] = np.polyval(self.r0, self.result['Probit'])
        self.result = self.result.sort_values(by='RSR回归', ascending=False)
    
    def __rank_result_print(self):
        print('等级\tProbit\t\tRSR\t\t\t分档排序结果')
        print('上\t>=%g\t\t>=%f\t\t%s' % (self.threshold[-1], np.polyval(self.r0, self.threshold[-1]),
                                         sorted(list(self.result[self.result['RSR回归'] >= np.polyval(self.r0,
                                                                         self.threshold[-1])].index))))
        for i in range(len(self.threshold) - 1):
            print('中\t%g~%g\t%f~%f\t%s' % (self.threshold[i], self.threshold[i + 1],
                                           np.polyval(self.r0, self.threshold[i]),
                                           np.polyval(self.r0, self.threshold[i + 1]),
                                           sorted(list(self.result[(self.result['RSR回归'] > np.polyval(self.r0, self.threshold[i])) & (self.result['RSR回归'] < np.polyval(self.r0, self.threshold[i + 1]))].index))))
    
    #输出RSR分析表格
    def __save_excel(self):
        file_path = os.path.join(os.path.expanduser("~"), 'Desktop') + '\\RSR 分析结果报告.xlsx'
        excel_writer = pd.ExcelWriter(file_path)
        self.result.to_excel(excel_writer, '分档排序结果')
        self.distribution.to_excel(excel_writer, 'RSR分布表', index=False)
        excel_writer.save()
        
    #封装初始化函数
    def __Initialize(self):
        '''初始化函数，执行一次'''
        self.__check_w()
        self.__isbenefit()
        self.__make_up()
        self.__calculate_the_rank_sum_ratio()
        self.__distribution_rsr_table()
        self.__regression()
        self.__fit_probit()
        self.__save_excel()
        self.__rank_result_print()
     
        
    def update(self, **kwargs):
        '''用于调参'''
        self.w = list(self.w)
        if kwargs.get('benefit_index'):
            self.benefit_index = kwargs['benefit_index']
        if kwargs.get('w'):
            self.w = kwargs['w']
        if kwargs.get('full_rank'):
            self.full_rank = kwargs['full_rank']
        if kwargs.get('threshold'):
            self.threshold = kwargs['threshold']
        self.result = pd.DataFrame()
        self.shape = self.data.shape
        self.ascending = [0] * self.data.shape[1]
        self.__Initialize()
        
    
if __name__ == '__main__':
    
    #读取数据
    data = pd.read_excel(r"C:\Users\Ranger -speed\Desktop\wangwenwen.xlsx")
#    data = pd.read_excel(r"C:\Users\Ranger -speed\Desktop\wangwenwen.xlsx", sheetname=1)
#    data = pd.read_excel(r"C:\Users\Ranger -speed\Desktop\wangwenwen.xlsx", sheetname=2)
#    data = pd.read_excel(r"C:\Users\Ranger -speed\Desktop\wangwenwen.xlsx", sheetname=3)
    data1 = data.iloc[:, 0:7]
    print(data1.shape[1])
    data2 = data.iloc[:, 7:14]
    data3 = data.iloc[:, 14:18]
    
    #读取权重
    W = pd.read_excel(r"C:\Users\Ranger -speed\Desktop\沿岸北上型权重.xlsx")
#    W = pd.read_excel(r"C:\Users\Ranger -speed\Desktop\沿海北上型权重.xlsx")
#    W = pd.read_excel(r"C:\Users\Ranger -speed\Desktop\正面登陆型权重.xlsx")
#    W = pd.read_excel(r"C:\Users\Ranger -speed\Desktop\南部登陆型权重.xlsx")
    W1 = W.iloc[0, 0:7]
    print(W1, len(W1))
    W2 = W.iloc[0, 8:15]
    W3 = W.iloc[0, 16:20]
    
#    data = pd.DataFrame({'产前检查率': [99.54, 96.52, 99.36, 92.83, 91.71, 95.35, 96.09, 99.27, 94.76, 84.80],
#                         '孕妇死亡率': [60.27, 59.67, 43.91, 58.99, 35.40, 44.71, 49.81, 31.69, 22.91, 81.49],
#                         '围产儿死亡率': [16.15, 20.10, 15.60, 17.04, 15.01, 13.93, 17.43, 13.89, 19.87, 23.63]},
#                        index=list('ABCDEFGHIJ'), columns=['产前检查率', '孕妇死亡率', '围产儿死亡率'])
    RSR(data1, w=W1)