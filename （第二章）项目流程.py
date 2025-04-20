# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 19:31:23 2018

@author: 游侠-Speed
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

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

df = pd.read_excel(r'C:\Users\Shanks\Desktop\第二版实证随机森林插补改良-实证.xlsx', sheet_name=1, index_col=0)
print('读取>>>>>>>>>>>>>>>>>>>>>>>> \n', df)
df_describe = df.describe()
print(df_describe)


#各变量直方图
df.hist(bins=50, figsize=(20,15))
#plt.xlabel('成交量(股)')
#plt.ylabel('频数')
#plt.title('成交量分布直方图')
plt.show()

#各变量散点图
scatter_matrix(df, figsize=(20, 15))
plt.show()


#pearson热度图
corr_matrix = df.corr()
corr_matrix["直接经济损失"].sort_values(ascending=False)

colormap = plt.cm.RdBu
plt.figure(figsize=(18,18))
plt.title('Pearson Correlation Coefficients', y=1.05, size=15)
sns.heatmap(corr_matrix.astype(float), linewidths=0.1, vmax=1.0, square=True, \
            cmap=colormap, linecolor='white', annot=True, fmt='.2f')
plt.show()


#纯随机抽样
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) 



#分层抽样A
#from sklearn.model_selection import StratifiedShuffleSplit
#
#df["income_cat"] = np.ceil(df["倒塌房屋.间."] / 1.5)
#df["income_cat"].where(df["income_cat"] < 5, 5.0, inplace=True)
#
#split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#for train_index, test_index in split.split(df, df["income_cat"]):
#    strat_train_set = df.loc[train_index]
#    strat_test_set = df.loc[test_index]
#
#for set in (strat_train_set, strat_test_set):
#    set.drop(["income_cat"], axis=1, inplace=True)

df1 = train_set.drop('类别', axis=1)
df1_labels = train_set["类别"].copy()
num_attribs = list(df1)

#把文本数据转化成数字
#from sklearn.preprocessing import LabelEncoder
#encoder = LabelEncoder()
#housing_cat = housing["ocean_proximity"]
#housing_cat_encoded, housing_categories = housing_cat.factorize()
#housing_cat_encoded[:10]
'''
#标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X) #x为二维数组
#归一化
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)#因为这地方用了fit_transform，所以可以再对测试集转化
X_test_minmax = min_max_scaler.transform(X_test)
'''



#流水线数据预处理，先清洗，再标准化
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

from sklearn.pipeline import FeatureUnion

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
    
df1_prepared = num_pipeline.fit_transform(df1)
#得到的是多维数组


#KNN最近邻算法 分类算法，在这里插一下，补充之前没有用sklearn提供的方法
from sklearn import neighbors  
knn = neighbors.KNeighborsClassifier()
knn.fit(X, labels)
knn.predict()
plt.scatter()
plt.text()

#Kmeans聚类算法，在这插一下，补充之前没有用sklearn提供的方法
'''
簇数量需先给定
不适用非线性边界
计算较慢
'''
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
# make_blobs聚类数据生成器

x, y_true = make_blobs(n_samples=300,
                       centers=4,
                       cluster_std=0.5,  #方差一致为0.5
                       random_state=0)
print(x[:5])
print(y_true[:5])

kmeans = KMeans(n_clusters=4)
kmeans.fit(x)
y_kmeans = kmeans.predict(x) #一维类别数组
centroids = kmeans.cluster_centers_ #中心点坐标,有顺序
plt.scatter(x[:,0], x[:,1], c=y_kmeans, cmap='Dark2', s=50, alpha=0.5, marker='x')
plt.scatter(centroids[:,0], centroids[:,1], c=[0,1,2,3], cmap='Dark2', s=70, alpha=0.7, marker='o')


from sklearn.metrics import mean_squared_error

#线性回归
from sklearn.linear_model import LinearRegression #线性回归时检查下变量相关性
lin_reg = LinearRegression()
lin_reg.fit(df1_prepared, df1_labels)  #model.score可知R方系数
df1_predictions = lin_reg.predict(df1_prepared)
lin_mse = mean_squared_error(df1_labels, df1_predictions)
lin_rmse = np.sqrt(lin_mse)
print('>>>>>>>>>>>>>>>>>>>>>>>>线性回归的均方误差为： \n', lin_rmse)

#树回归
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(df1_prepared, df1_labels)
df1_predictions = tree_reg.predict(df1_prepared)
tree_mse = mean_squared_error(df1_labels, df1_predictions)
tree_rmse = np.sqrt(tree_mse)
print('>>>>>>>>>>>>>>>>>>>>>>>>树回归的均方误差为： \n', tree_rmse)

#树分类

from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier()
tree_clf.fit(df1_prepared, df1_labels)
df1_predictions = tree_clf.predict(df1_prepared)
tree_mse = mean_squared_error(df1_labels, df1_predictions)
tree_rmse = np.sqrt(tree_mse)
print('>>>>>>>>>>>>>>>>>>>>>>>>树分类的均方误差为： \n', tree_rmse)

#随机森林
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(df1_prepared, df1_labels)
df1_predictions = forest_reg.predict(df1_prepared)
forest_mse = mean_squared_error(df1_labels, df1_predictions)
forest_rmse = np.sqrt(forest_mse)
print('>>>>>>>>>>>>>>>>>>>>>>>>随机森林的均方误差为： \n', forest_rmse)

#支持向量机
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVR
svr_reg = LinearSVR(epsilon=1.5)
svr_poly_reg = SVR(kernel='poly')
svr_rbf_reg = SVR(kernel='rbf')
#svr_nu_reg = NuSVR()
svr_reg.fit(df1_prepared, df1_labels)
df1_predictions = svr_reg.predict(df1_prepared)
svr_mse = mean_squared_error(df1_labels, df1_predictions)
svr_rmse = np.sqrt(svr_mse)
print('>>>>>>>>>>>>>>>>>>>>>>>>支持向量机的均方误差为： \n', svr_rmse)


#交叉验证
from sklearn.model_selection import cross_val_score
scores = cross_val_score(svr_reg, df1_prepared, df1_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print("Scores:", scores)
print("Mean:", tree_rmse_scores.mean())
print("Standard deviation:", tree_rmse_scores.std())


#保存试验过的模型，读取
from sklearn.externals import joblib
joblib.dump(final_model, "my_model.pkl")
my_model_loaded = joblib.load("my_model.pkl")



#找到最佳模型后再用网格搜索对模型超参数进行微调
from sklearn.model_selection import GridSearchCV

#param_grid = [
#    {'n_estimators': [3, 10, 30], 'max_features': [2, 3]},
#    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3]},
#  ] 

param_grid = [
        {'C':[0.1, 0.3, 1, 3, 10, 30, 100, 300], 
         'epsilon':[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]}
        ]  #线性SVR没有核函数参数，限制了只能使用线性核函数
grid_search = GridSearchCV(svr_reg, param_grid, cv=10,
                           scoring='neg_mean_squared_error')
grid_search
grid_search.fit(df1_prepared, df1_labels)
print(grid_search.best_params_)  #参数最佳组合
print(grid_search.best_estimator_)  #最佳估计器

###################网格搜索画图#############################
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
#ax = Axes3D(fig)
#保持xx yy z大小一致，位置对应
X = [0.1, 0.3, 1, 3, 10, 30, 100, 300]
Y = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
Z = np.sqrt(-grid_search.cv_results_['mean_test_score']).reshape(8,8).T
#X = [x['C'] for x in grid_search.cv_results_['params']]
#Y = [y['epsilon'] for y in grid_search.cv_results_['params']]
XX, YY = np.meshgrid(X, Y)
surf = ax.plot_surface(XX, YY, Z, cmap='rainbow', linewidth=0, antialiased=True)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('C')
ax.set_ylabel('epsilon')
ax.set_zlabel('RMSE')
ax.set_title('SVR参数选择结果图(3D视图)[GridSearchMethod]\n Best c=300,epsilon=0.3,RMSE=117.527687191')
#ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()
###################网格搜索画图#############################

#查看评估分数
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)


#分析最佳模型和它们变量的误差
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
attributes = num_attribs
sorted(zip(feature_importances,attributes), reverse=True)


#用测试集评估系统
final_model = grid_search.best_estimator_
#final_model = forest_reg
df_all = pd.read_excel(r'C:\Users\Shanks\Desktop\沪深300.xlsx', index_col=0, sheet_name=7)

X_test = df_all

#X_test = test_set.drop("一周后的开盘价", axis=1)
#y_test = test_set["一周后的开盘价"].copy()

X_test_prepared = num_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print('>>>>>>>>>>>>>>测试集均方误差为： \n', final_rmse)













#x_axis = np.arange(0, 8)
#x__axis = np.arange(8, 10)
#
##分样本并进行归一化
#y = np.array(df.iloc[:, -1])
#X = np.array(df.iloc[:, :-1])
#
#X_train = X[0:8]
#y_train = y[0:8]
#
#X_test = X[8:]
#y_test = y[8:]
#
#X_train = StandardScaler().fit_transform(X_train)
#X_test = StandardScaler().fit_transform(X_test)
#
##调参
#grid = GridSearchCV(SVR(kernel='rbf'),
#                    param_grid={"C":[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
#                    "gamma": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
#                    "epsilon": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]},
#                     )
#grid.fit(X_train, y_train)
#print("The best parameters are {} with a score of {}".format(grid.best_params_, grid.best_score_))
#
#
##开始拟合和测试
#rbf_svr=SVR(kernel='rbf', C=1000,  gamma=0.2)
#rbf_svr.fit(X_train,y_train)
#
#y_predict = rbf_svr.predict(X_train)
#y__predict =rbf_svr.predict(X_test)
#
#
#fig, ax = plt.subplots(2)
#ax[0].plot(x_axis, y_train, "b", label="original")
#ax[0].plot(x_axis, y_predict, "r", label="predict")
#ax[0].set(title="train")
#ax[0].legend()
#
#ax[1].plot(x__axis , y_test, "b", label="original")
#ax[1].plot(x__axis , y__predict, "r", label="predict")
#ax[1].set(title="test")
#ax[1].legend()
#
#
#y1 = np.array([y_test])
#y2 = np.array([y__predict])
#
#y3 = np.sum(((y1 - y2)**2) / y1.shape[0])
#print("均方误差>>>>>>>>>", y3)


#y4 = np.concatenate([y1.reshape(6,1), y2.reshape(6,1)], axis=1)
#y4 = pd.DataFrame(y4.reshape(6, 2), columns=["A", "B"])
#print(y4.corr("spearman"))


#Kappa 检验
#class Kappa: 
#                  
#    def metrics(self,pre1,pre2,classN=14):
#        k1=np.zeros([classN,])
#        print(k1)
#        k2=np.zeros([classN,])
#        kx=np.zeros([classN,])
#        n=np.size(pre1)
#        for i in range(n):
#            p1=pre1[i]
#            p2=pre2[i]
#            k1[p1-1]=k1[p1-1]+1
#            k2[p2-1]=k2[p2-1]+1
#            if p1==p2:
#                kx[p1-1]= kx[p1-1]+1
#        
#        pe=np.sum(k1*k2)/n/n
#        pa=np.sum(kx)/n
#        kappa=(pa-pe)/(1-pe)
#        
#        return kappa
#   
#    
#y1 = y1.reshape(3)
#y2 = y2.reshape(3)
#
#print(y1)
#kappa = Kappa()
#kappa_test = kappa.metrics(y1.astype(int), y2.astype(int))
 




