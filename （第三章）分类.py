# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:43:51 2019

@author: Shanks
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn import datasets, svm
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_openml

#mnist数据集载入
mnist = fetch_openml('MNIST original')
mnist
X, y = mnist['data'], mnist['target']
some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')
plt.show()
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#风暴潮数据载入
df = pd.read_excel(r'C:\Users\Shanks\Desktop\第二版实证随机森林插补改良-用来筛选.xlsx', sheet_name=1, index_col=0)
print('读取>>>>>>>>>>>>>>>>>>>>>>>> \n', df)
df_describe = df.describe()
print(df_describe)

####风暴潮分训练集测试集
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
df1 = train_set.drop('类别', axis=1)
df1_labels = train_set["类别"].copy()
num_attribs = list(df1)

#洗牌（非必须项）
import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#创建目标向量
##1 示例
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
##2 随机森林风暴潮
df1_labels_low = (df1_labels == 'low')#二分类就用这个，多分类就用df1_labels

#两个分类器
##1 sgd分类器
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])
##2 随机森林分类器
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(df1, df1_labels) #df1_prepared是从项目流程里数据预处理过来的
forest_clf.predict(df1)


#两种交叉验证
#1
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
#2
from sklearn.model_selection import cross_val_score
df1_scores = cross_val_score(forest_clf, df1, df1_labels, cv=10, scoring='accuracy')#返回评估分数
np.mean(df1_scores)

#康一康特征重要性
for name, score in zip(df1.columns.values.tolist(), forest_clf.feature_importances_):
    print(name, score)

#混淆矩阵评估分类器性能 风暴潮
from sklearn.model_selection import cross_val_predict
df1_prepared_pred = cross_val_predict(forest_clf, df1, df1_labels, cv=10)#返回预测值

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(df1_labels, df1_prepared_pred)


#有了混淆矩阵就可以计算精度和召回率
from sklearn.metrics import precision_score, recall_score
precision_score(df1_labels_low, df1_prepared_pred)
recall_score(df1_labels_low, df1_prepared_pred)
#计算F1分数
from sklearn.metrics import f1_score
f1_score(df1_labels, df1_prepared_pred)


#调整阈值
y_scores = sgd_clf.decision_function([some_digit])
y_scores
threshold = 200000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
#返回训练集中所有样本的决策分数
y_scores = cross_val_predict(forest_clf, df1_prepared, df1_labels, cv=3, method='decision_function')#sgd分类器用'decision_function'
df1_proba_pred = cross_val_predict(forest_clf, df1_prepared, df1_labels, cv=5, method='predict_proba')#随机森林用'predict_proba'先获取概率
df1_proba_pred1 = df1_proba_pred[:, 1] #但是绘制ROC曲线需要的是分数值而不是概率值，一个简单的解决方案是：直接用正类的概率作为分数值


#计算所有可能的精度和召回率，精度和召回率相对于阈值的函数图
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
    

#绘制ROC曲线
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(df1_labels, df1_proba_pred1)
def plot_roc_curve(fpr, tpr, label='Random Forest'):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Ture Positive Rate')
plot_roc_curve(fpr, tpr)
plt.show()
#计算曲线下面积
from sklearn.metrics import roc_auc_score    
roc_auc_score(df1_labels_low, df1_prepared_pred)


#已经有了一个有潜力的模型，现在希望进行改进，方法之一就是分析其错误类型
#先缩放一下
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train)
#首先查看混淆矩阵
conf_mx = confusion_matrix(df1_labels, df1_prepared_pred)
conf_mx
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
#计算错误率
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)#用0填充对角线
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


#多标签分类
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([some_digit])

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
f1_score(y_train, y_train_knn_pred, average='macro')
    

