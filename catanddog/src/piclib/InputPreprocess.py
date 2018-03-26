#=========================================================================
#copy right 2018  tingwu.all rights reserved
#Date :  2016.03.02
#function: jpg FileReader
#Writer: Tingwu
#email: 18510665908@163.com
#CSVReader.py
#=========================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import os
import common

TRAINING_STEPS = 10
IMAGE_SRC_PATH = "/home/zhangtx/ml/catanddog/data/train/*jpg"
files = tf.train.match_filenames_once(IMAGE_SRC_PATH)
filename_queue = tf.train.string_input_producer(files, shuffle=False)
init = tf.global_variables_initializer()

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64),
    })
image = features['image']
label = features['label']
height = features['height']
width = features['width']
channels = features['channels']

decode_image = tf.decode_raw(image, tf.uint8)
decode_image.set_shape((height, width, channels))

image_size = 299

distorted_image = common.distort_color(decode_image, color_ordering=1)


min_after_deque = 10000
batch_size = 100
capacity = min_after_deque + 3 * batch_size

image_batch, label_batch = tf.train.shuffle_batch(
    [distorted_image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_deque)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(TRAINING_STEPS):
        sess.run(label_batch)
        sess.run(image_batch)

    coord.request_stop()
    coord.join(threads)



