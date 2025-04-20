# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 20:48:09 2019

@author: Shanks
"""
import numpy as np


############################无量纲化########################################


from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler #标准化
from sklearn.preprocessing import MinMaxScaler #缩放
from sklearn.preprocessing import Normalizer #归一化
from sklearn.preprocessing import Binarizer #二值化
from sklearn.preprocessing import OneHotEncoder #哑编码


iris = load_iris()

iris.data  #二维数组
iris.target #一维数组
 
StandardScaler().fit_transform(iris.data) #按列标准化，前提是符合正态分布，标准化后，转换为标准正态分布
MinMaxScaler().fit_transform(iris.data)  #按列进行区间缩放 [0, 1]
Normalizer().fit_transform(iris.data)  #按行进行归一化，使得行向量模为1，
                                #其目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准
Binarizer(threshold=3).fit_transform(iris.data) #二值化，阈值设置为3，返回值为二值化后的数据
OneHotEncoder().fit_transform(iris.target.reshape((-1,1))) #哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据


############################无量纲化########################################

############################特征选择########################################

#第一大类Filter：过滤法，按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征。

#方差选择法
from sklearn.feature_selection import VarianceThreshold #使用方差选择法，先要计算各个特征的方差，然后根据阈值，选择方差大于阈值的特征。
VarianceThreshold(threshold=3).fit_transform(iris.data) #方差选择法，返回值为特征选择后的数据，参数threshold为方差的阈值

#相关系数法
from sklearn.feature_selection import SelectKBest #使用相关系数法，先要计算各个特征对目标值的相关系数以及相关系数的P值。
from scipy.stats import pearsonr
SelectKBest(lambda X, Y: np.array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)#选择K个最好的特征，返回选择特征后的数据
                                                                                    #第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
#卡方检验                                                                             #参数k为选择的特征个数
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2  #这个统计量的含义简而言之就是自变量对因变量的相关性
SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)#选择K个最好的特征，返回选择特征后的数据

#互信息
from sklearn.feature_selection import SelectKBest
from minepy import MINE  #经典的互信息也是评价定性自变量对定性因变量的相关性的，为了处理定量数据，最大信息系数法被提出
                            #由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)
SelectKBest(lambda X, Y: np.array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)#选择K个最好的特征，返回特征选择后的数据

#第二大类 Wrapper：包装法，根据目标函数（通常是预测效果评分），每次选择若干特征，或者排除若干特征。

#递归特征消除法， 递归消除特征法使用一个基模型来进行多轮训练，每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练。
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)#递归特征消除法，返回特征选择后的数据
                                                                                                 #参数estimator为基模型
                                                                                                 #参数n_features_to_select为选择的特征个数
#第三大类 Embedded：嵌入法，先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。类似于Filter方法，但是是通过训练来确定特征的优劣。
     #未完待续.......   https://www.zhihu.com/question/28641663/answer/146730530


############################特征选择########################################


############################降维########################################
from sklearn.decomposition import PCA
#主成分分析法，返回降维后的数据
#参数n_components为主成分数目
PCA(n_components=2).fit_transform(iris.data)


from sklearn.lda import LDA
#线性判别分析法，返回降维后的数据
#参数n_components为降维后的维数
LDA(n_components=2).fit_transform(iris.data, iris.target)


