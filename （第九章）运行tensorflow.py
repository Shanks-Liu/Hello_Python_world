# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:43:05 2020

@author: Shanks
"""

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


#一个简单的运行示例
import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x*x*y + y + 2

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    
#线性回归的例子
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
thete = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)  #正规方程

with tf.Session() as sess:
    theta_value = theta.eval()
    

#梯度下降
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name='theta')
y_pred =  tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
gradients = tf.gradients(mse, [theta])[0] #自动计算微分
training_op = tf.assign(theta, theta - learning_rate * gradients) #梯度迭代
######梯度下降优化器，替换上两行##############################################
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)
#############################################################################

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print('Epoch', epoch, 'MSE = ', mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()


#保存和恢复模型
saver = tf.train.Saver()
            save_path = saver.save(sess, '/tmp/my_model.ckpt')
    save_path = saver.save(sess, '/tmp/my_model_final.ckpt')

saver.restore(sess, '/tmp/my_model_final.ckpt')

#批量训练
learning_rate = 0.01
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()







#用TensorBoard来可视化图和训练曲线
'''
每次运行程序时，都要指定一个不同的目录，就用时间戳来命名日志文件夹
'''
#程序最开始部分
from datetime import datetime
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = '{}/run-{}/'.format(root_logdir, now)
#放到构造期末尾
mse_summary = tf.summary.scaler('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

#前两个完成后，在循环中加入求值mse_sunmary节点
for batch_index in range(n_batched):
    X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
    if batch_index % 10 == 0:
        summary_str = mse_summary.eval(feed_dict={X:X_batch, y:y_batch})
        step = epoch * n_batches + batch_index
        file_writer.add_summary(summary_str, step)
    sess.run(training_op, feed_dict={X:X_batch, y:y_batch})