# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 21:49:00 2019

@author: Shanks
"""

'''
本章会讨论适合线性数据的线性回归和适合非线性数据的多项式回归
'''
import numpy as np


#两种完全不同的训练模型的方法
#1.标准方程 计量经济学公式
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

#2.梯度下降，应用时需要保证特征值大小比例差不多，standardscaler
#2.1 批量梯度下降
eta = 0.1
n_iterations = 1000
m = 100
theta = np.random.rand(2,1)
for iterations in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta *gradients
#2.2 随机梯度下降
n_epochs = 50
t0, t1 = 5, 50
def learning_schedule(t):
    return t0 / (t + t1)
theta = np.random.randn(2,1)
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
#2.3 调用SGD执行线性回归
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())

######多项式回归
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
#调用,将每个特征的平方作为新特征加入训练集
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]
X_poly[0]
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()   #然后用线性拟合
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_


#学习曲线：除了交叉验证，另外一种判断过拟合的方法
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), 'r-+', linewidth=2, label='train')
    plt.plot(np.sqrt(val_errors), 'b-', linewidth=3, label='val')
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)

#减少过拟合的方法
sgd = SGDRegressor(penalty='l2')#岭回归（l2范数的一半）
sgd_reg.fit(X, y,ravel())
sgd_reg.predict([[1.5]])

from sklearn.linear_model import Lasso #Lasso回归(l1范数)
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])
        
from sklearn.base import clone#早期停止法
sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None, 
                       learning_rate='constant', eta0=0.0005)
minimum_val_error = float('inf')
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train)
    y_val_predict = sgd_reg_predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val_predict, y_val)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        