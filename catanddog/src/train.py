# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import inferency
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import piclib.Transpose
import piclib.common
import time

EXAMPLES_NUM = 16000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.003
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.001
TRAINING_STEPS = 300000
MOVING_AVERAGE_DECAY = 0.99


MODEL_SAVE_PATH = "/home/zhangtx/ml/catanddog/model"
MODEL_NAME = "model.ckpt"

def train(filename):
    x = tf.placeholder(tf.float32, [BATCH_SIZE,
                                    inferency.IMAGE_SIZE,
                                    inferency.IMAGE_SIZE,
                                    inferency.NUM_CHANNELS], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, inferency.OUTPUT_NODE], name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = inferency.interfence(x, True, regularizer)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())

    cross_entroy = tf.nn.sparse_softmax_cross_entropy_with_logits(
         logits=y, labels=tf.argmax(y_, 1))
    cross_entroy_mean = tf.reduce_mean(cross_entroy)

    loss = cross_entroy_mean + tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar("loss", loss)

    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            EXAMPLES_NUM/BATCH_SIZE,
            LEARNING_RATE_DECAY)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("../logs/", tf.get_default_graph())
    image_batch, label_batch, pixes_batch = piclib.Transpose.make_batch(filename,BATCH_SIZE)
    coord = tf.train.Coordinator()

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    merged_summary = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(TRAINING_STEPS):
            raw_image, raw_label, raw_pixes = sess.run([image_batch, label_batch, pixes_batch])
            reshaped_xs = np.reshape(raw_image, (BATCH_SIZE,
                                                 inferency.IMAGE_SIZE,
                                                 inferency.IMAGE_SIZE,
                                                 inferency.NUM_CHANNELS))
            reshaped_ys = piclib.common.oneShot(raw_label, 2)
            _, loss_value, step, summary = sess.run([train_op, loss, global_step,merged_summary],
                                            feed_dict={x: reshaped_xs, y_: reshaped_ys})

            if i % 50 == 0:
                print("After %d training steps,loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
            writer.add_summary(summary, step)
            time.sleep(1)
    coord.request_stop()
    coord.join(threads)

def main(argv=None):
    # train(["../data/dest/output.tfrecords.blur.train1",
    #        "../data/dest/output.tfrecords.blur.train2",
    #        "../data/dest/output.tfrecords.flip.train1",
    #        "../data/dest/output.tfrecords.flip.train2",
    #        "../data/dest/output.tfrecords.normal.train1",
    #        "../data/dest/output.tfrecords.normal.train2",
    #        "../data/dest/output.tfrecords.rotate.train1",
    #        "../data/dest/output.tfrecords.rotate.train2",
    #        "../data/dest/output.tfrecords.tb.train1",
    #        "../data/dest/output.tfrecords.tb.train2",
    #        "../data/dest/output.tfrecords.lr.train1",
    #        "../data/dest/output.tfrecords.lr.train2",
    #        "../data/dest/output.tfrecords.rotate90.train1",
    #        "../data/dest/output.tfrecords.rotate90.train2"
    #        ])
    train([
           "../data/dest/output.tfrecords.normal.train1",
           "../data/dest/output.tfrecords.normal.train2"
           ])
if __name__ == '__main__':
    tf.app.run()












