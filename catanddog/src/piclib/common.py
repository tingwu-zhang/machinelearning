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
import tensorflow as tf
import numpy as np
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return tf.clip_by_value(image, 0.0, 1.0)

def oneShot(label_batch,cat_num):
    num_labels = label_batch.shape[0]
    index_offset = np.arange(num_labels) * cat_num
    num_labels_hot = np.zeros((num_labels, cat_num))
    num_labels_hot.flat[index_offset+label_batch.ravel()] = 1.0
    return num_labels_hot