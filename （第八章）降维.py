# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 21:01:16 2020

@author: Shanks
"""

'''
两种主要的数据降维方法：投影  流形学习
三种数据降维技术： PCA； Kernal PCA；  LIE
'''

from sklearn.decomposition import PCA

pca = PCA(n_components=2)  #会自动处理数据集中，先提取两个主成分
X2D = pca.fit_transform(X)  #然后转化,就是数据集矩阵和主成分矩阵的点积

pca.components_.T[:,0]  #查看第一个主成分
print(pca.explained_variance_ratio_)  #查看每个主成分的方差解释率
print(pca.explained_variance_)  
print(pca.components_.T)  #主成分矩阵 VT
print(pca.n_components_) 

#找到方差解释率相加>0.95的主成分数量
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1  #然后设置n_components = d，再运行PCA

#比上个方法更方便
pca = PCA(n_components=0.95)  #浮点数表示希望保留的方差比
X_reduced = pca.fit_transform(X)

#逆转换：先压缩再解压缩
pca = PCA(n_components=154)
X_minist_reduced = pca.fit_transform(X_mnist)
X_minist_recoverd = pca.inverse_transform(X_mnist_reduced)

#核主成分分析
from sklearn.decomposition import kernelPCA

rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)


#KPCA本身是一种无监督的学习算法
#1.PCA作为监督式学习任务的准备步骤，找到使任务性能最佳的核函数和调整超参数
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
        ('kpca', KernelPCA(n_components=2)),
        ('log_reg', LogisticRegression())
                ])

param_grid = [{
        'kpca_gamma': np.linspace(0.03, 0.05, 10),
        'kpca_kernel': ['rbf', 'sigmoid']
        }]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X,y)

print(grid_search.best_params_)

#2.完全不受监督方法，选择使重建误差最低的核和超参数
rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.0433,
                    fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)

from sklearn.metrics import mean_squared_error
mean_squared_error(X, X_preimage)


#局部线性嵌入（LIE），是一种流形学习技术
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)






