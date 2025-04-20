# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:06:23 2019

@author: Shanks
"""
'''
决策树完全不需要进行特征缩放或集中
'''
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
import graphviz

df = pd.read_excel(r'C:\Users\Shanks\Desktop\第二版实证随机森林插补改良-用来决策树.xlsx', sheet_name=1, index_col=0)
print('读取>>>>>>>>>>>>>>>>>>>>>>>> \n', df)
df_describe = df.describe()
print(df_describe)

###########在项目流程里再把数据处理一遍###########树模型不需要缩放预处理
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) 

df1 = train_set.drop('类别', axis=1)
df1_labels = train_set["类别"].copy()
num_attribs = list(df1)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
    ])
    
df1_prepared = num_pipeline.fit_transform(df1)
###########处理完毕，得到数组####

tree_clf = DecisionTreeClassifier()

#用网格搜索对决策树模型超参数进行微调
#max_depth（树的深度）
#max_leaf_nodes（叶子结点的数目）
#max_features（最大特征数目）
#min_samples_leaf（叶子结点的最小样本数）
#min_samples_split（中间结点的最小样本数）
#min_weight_fraction_leaf（叶子节点的样本权重占总权重的比例）
#min_impurity_split（最小不纯净度）也可以调整
from sklearn.model_selection import GridSearchCV

max_depth = range(1, 8, 1)
max_leaf_nodes = range(1, 20, 2)
min_samples_leaf = range(5, 20, 2)
tuned_parameters = dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

grid_search = GridSearchCV(tree_clf, tuned_parameters,cv=8)
grid_search.fit(df1_prepared, df1_labels)

print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_)) #最高评分
print(grid_search.best_estimator_)  #最佳估计器

#查看评估分数
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)


###################网格搜索画图#############################
import numpy as np
from matplotlib import pyplot as plt
plt.figure(figsize=(10,8))

test_means = grid_search.cv_results_[ 'mean_test_score' ]
test_scores = np.array(test_means).reshape(len(max_depth), len(min_samples_leaf))
 
for i, value in enumerate(max_depth):
    plt.plot(min_samples_leaf, test_scores[i], label= 'test_max_depth:' + str(value))
 
plt.legend()
plt.xlabel('min_samples_leaf' )                                                                                                      
plt.ylabel('accuray' )
plt.show()
###################网格搜索画图#############################

final_model = grid_search.best_estimator_

###############可视化############
export_graphviz(
        final_model,
        out_file=r'C:\Users\Shanks\Desktop\风暴潮_final_tree.dot',
        feature_names=df1.columns.values.tolist(),
        class_names=df1_labels.tolist(),
        rounded=True,
        filled=True)

with open(r'C:\Users\Shanks\Desktop\风暴潮_final_tree.dot',encoding='utf-8') as fj:
    source=fj.read()

dot=graphviz.Source(source)
dot.view()
###############可视化############

X_test = test_set.drop("类别", axis=1)
y_test = test_set["类别"].copy()

X_test_prepared = num_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

from sklearn.metrics import accuracy_score
print ('测试集准确率：', accuracy_score(y_test, final_predictions))

#单独拿个样本估算预测，得到分到每个类比的概率数组
tree_clf.predict_proba([[5, 1.5]])
tree_clf.predict([[5, 1.5]])

#回归
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y) 