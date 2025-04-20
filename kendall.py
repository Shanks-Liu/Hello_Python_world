# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 19:35:55 2018

@author: 游侠-Speed
"""
import numpy as np
import pandas as pd

########kendall一致性检验##########
file = r'C:\\Users\Shanks\desktop\a.xlsx'
df = pd.read_excel(file, sheet_name=1)
print('>>>>>>>>>>>>>>>', df)
index = df['变量层']
del df['变量层']
m, n = np.shape(df)
Ri = np.sum(df, axis=1)
W = np.sum(Ri ** 2) - ((1 / m) * (np.sum(Ri) ** 2)) 
print('最终结果为：' + str(W) + ', 随后查阅Kendall一致性系数的临界值表来确定是否拒绝原假设')
########kendall一致性检验##########




#Kappa一致性检验
import numpy as np
class Kappa: 
                  
    def metrics(self,pre1,pre2,classN=11):
        k1=np.zeros([classN,])
        k2=np.zeros([classN,])
        kx=np.zeros([classN,])
        n=np.size(pre1)
        for i in range(n):
            p1=pre1[i]
            p2=pre2[i]
            k1[p1-1]=k1[p1-1]+1
            k2[p2-1]=k2[p2-1]+1
            if p1==p2:
                kx[p1-1]= kx[p1-1]+1
        
        pe=np.sum(k1*k2)/n/n
        pa=np.sum(kx)/n
        kappa=(pa-pe)/(1-pe)
        
        return kappa
    
kappa = Kappa()
kappa_test = kappa.metrics(np.array([8,9,6,7,4,3,2,11,5,1,10]),np.array([9,6,8,7,4,3,2,10,5,1,11]))
print(kappa_test)




#spearman 等级相关检验
import numpy as np
import pandas as pd
df = pd.DataFrame({'海域':np.array([2,1,2,0,1,1,1,1,0,2,2]),
     '陆域':np.array([2,0,2,1,2,1,2,0,1,0,0])})
print(df.corr("spearman"))


#Kendall Tau相关系数
df = pd.DataFrame({'海域':np.array([2,1,2,0,1,1,1,1,0,2,2]),
                   '陆域':np.array([2,0,2,1,2,1,2,0,1,0,0])})
print(df.corr('kendall'))  






