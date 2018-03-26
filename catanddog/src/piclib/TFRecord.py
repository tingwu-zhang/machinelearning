#=========================================================================
#copy right 2018  tingwu.all rights reserved
#Date :  2016.03.02
#function: TFRecord
#Writer: Tingwu
#email: 18510665908@163.com
#TFRecord.py
#=========================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data

import numpy as np

def __int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def __bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist = input_data.read_data_sets("../mnist/ministdata",dtype=tf.uint8, one_hot=True)

images = mnist.train.images
labels = mnist.train.labels

pixes = images.shape[1]

num_examples = mnist.train.num_examples

filename = "../../data/output.tfrecords"


print labels[0]
print np.argmax(labels[0])

writer = tf.python_io.TFRecordWriter(filename)

for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixes': __int64_feature(pixes),
        'label': __int64_feature(np.argmax(labels[index])),
        'image_raw': __bytes_feature(image_raw)
    }))
    writer.write(example.SerializeToString())

writer.close()