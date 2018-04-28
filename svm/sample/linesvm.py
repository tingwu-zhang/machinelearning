# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from pylab import mpl

FEATURE_NUM = 4
# mpl.rcParams['font.sans-serif'] = ['SimHei']

np.random.seed(1)
tf.set_random_seed(1)

iris = datasets.load_iris()
x_vals = iris.data
y_vals = np.array([1 if y ==0 else -1 for y in iris.target])

sample_num = len(x_vals)
train_num = int(np.round(sample_num * 0.8))
train_indices = np.random.choice(sample_num,train_num,replace=False)
print train_indices

test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
print test_indices

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

batch_size = 100
x_data = tf.placeholder(shape=[None, FEATURE_NUM], dtype=tf.float32)
y_data = tf.placeholder(shape=[None, 1],dtype=tf.float32)

W = tf.Variable(tf.random_normal([FEATURE_NUM, 1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weight"))
b = tf.Variable(tf.zeros([1, 1]), name="b", dtype=tf.float32)

y = tf.add(tf.matmul(x_data, W), b)
l2_norm = tf.reduce_mean(tf.square(W))
loss = l2_norm + tf.reduce_mean(tf.maximum(0.,1.0-y_data*y))

prediction = tf.sign(y)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_data), tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()
loss_vec = []
train_accuracy = []
test_accuracy = []
with tf.Session() as sess:
    sess.run(init)

    for i in range(500):
        rand_index = np.random.choice(len(x_vals_train),size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        sess.run(train_step,feed_dict={x_data:rand_x,y_data:rand_y})
        temp_loss = sess.run(loss,feed_dict={x_data:rand_x,y_data:rand_y})
        loss_vec.append(temp_loss)
        train_accuracy_temp = sess.run(accuracy,feed_dict={x_data:x_vals_train,y_data:np.transpose([y_vals_train])})
        train_accuracy.append(train_accuracy_temp)
        test_accuracy_temp = sess.run(accuracy,feed_dict={x_data:x_vals_test,y_data:np.transpose([y_vals_test])})
        test_accuracy.append(test_accuracy_temp)
        if (i+1)%10==0:
            print('step #'+str(i+1)+ ' w=' + str(sess.run(W)) + ' b='+ str(sess.run(b)))
plt.figure()
plt.plot(loss_vec)
plt.plot(train_accuracy)
plt.plot(test_accuracy)
plt.legend(['loss','train accu','test accu'])
plt.ylim(0.,1.)
plt.show()



