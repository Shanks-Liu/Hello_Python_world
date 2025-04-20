# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 21:04:21 2019

@author: Shanks
"""
#投票分类器，三种不同的分类器组成ijm,
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='hard'
        )
voting_clf.fit(X_train, y_train)


#sklearn提供的API,bagging 和 pasting
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        max_samples=100, bootstrap=True, n_jobs=-1
        )
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
#自动进行包外评估
bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        bootstrap=True, n_jobs=-1, oob_score=True)




###########随机森林###############
from sklearn.ensemble import RandomForestClassifier #回归就用RandomForestRegressor
rnd_clf = RandomForestClassifier(n_estimator=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)

#极端随机树，训练比常规随机森林快
from sklearn.ensemble import ExtraTreesClassifier

#估算一个特征在森林所有树上的平均深度，可以估算一个特征的重要程度,想快速了解什么是真正重要的特征，随机森林是一个非常便利的方法
from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris['data'], iris['target'])
for name, score in zip(iris['feature_names'], rnd_clf.feature_importances_):
    print(name, score)

#提升法（boosting，将几个弱学习器结合成一个强学习器，分为adaboost和gradient boosting）
#adaboost
from sklearn.ensemble import AdaboostClassifier
ada_clf = AdaboostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm='SAMME.R', learning_rate=0.5)
ada_clf.fit(X_train, y_train)

#gradient boosting
from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X, y)

y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X, y2)

y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X, y3)

y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
#以上是原理，下面是api
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X,y)

#用早期停止法找到最佳的决策树的数量
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y)
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_val, y_pred)
            for y_pred in gbrt.staged_predict(X_val)]
bes_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)

