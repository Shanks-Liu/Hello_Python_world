# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:34:12 2019
@author: Shanks
"""

'''
SVM特别适用于中小型复杂数据集的分类
大间隔分类
'''

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#线性SVM分类
iris = datasets.load_iris()
X = iris['data'][:, (2,3)]
y = (iris['target'] == 2).astype(np.float64)

svm_clf = Pipeline((
        ('scaler', StandardScaler()),
        ('linear_svc', LinearSVC(C=1, loss='hinge')),
        ))
svm_clf.fit(X_scaled, y)

#非线性SVM分类
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
polynomial_svm_clf = Pipeline((
        ('poly_features', PolynomialFeatures(degree=3)),
        ('scaler', StandardScaler()),
        ('svm_clf', LinearSVC(C=10, loss='hinge'))
        ))
polynomial_svm_clf.fit(X, y)

#多项式核
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline((
        ('scaler', StandardScaler()),
        ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
        ))
poly_kernel_svm_clf.fit(X, y)

rbf_kernel_svm_clf = Pipeline((
        ('scaler', StandardScaler()),
        ('svm_clf', SVC(kernel='rbf', degree=3, coef0=1, C=5))
        ))
poly_kernel_svm_clf.fit(X, y)#高斯核的好处：产生的结果跟添加了许多相似特征一样，但实际上并不需要添加

#SVM回归
from sklearn.svm import LinearSVR #没有核技巧
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X,y)


from sklearn.svm import SVR #有核技巧
svm_poly_reg = SVR(kernel='poly', degree=2, c=100, epsilon=0.1)
svm_poly_reg.fit(X, y)







