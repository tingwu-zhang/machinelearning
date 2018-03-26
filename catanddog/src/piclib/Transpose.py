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
import os
import matplotlib as plt
import numpy as np
from PIL import Image
from pylab import *
import ReadJpg

def __int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def __bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

PATH = "../../data/train/"
TEST_PATH = "../../data/test/"
TFRECORD_FILENAME = "../../data/dest/output.tfrecords"
SAME_SIZE_JPG = "../../data/dest/"
RESIZED_IMAGE_SIZE = 64

def encode_to_tfrecords(path, filename):
    writer_train = tf.python_io.TFRecordWriter(filename+".train")
    writer_validation = tf.python_io.TFRecordWriter(filename+".validation")

    for img_name in os.listdir(path):
        #label is decided by filename's prefix
        if img_name.startswith("dog"):
            label = 1
        else:
            label = 0

        img = Image.open(os.path.join(path, img_name))
        img = img.resize((RESIZED_IMAGE_SIZE, RESIZED_IMAGE_SIZE))
        example = tf.train.Example(features=tf.train.Features(feature={
            'pixes': __int64_feature(RESIZED_IMAGE_SIZE*RESIZED_IMAGE_SIZE),
            'label': __int64_feature(label),
            'image_raw': __bytes_feature(img.tobytes())
        }))
        print os.path.join(PATH, img_name) + ' is processed '

        chance = np.random.randint(1, 99)
        if chance < 12:
            writer_validation.write(example.SerializeToString())
        else:
            writer_train.write(example.SerializeToString())
    writer_train.close()
    writer_validation.close()

def decode_from_tfrecords(file):
    filename_queue = tf.train.string_input_producer([file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'pixes': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    img = tf.decode_raw(features['image_raw'], tf.uint8)

    img = tf.reshape(img, [RESIZED_IMAGE_SIZE, RESIZED_IMAGE_SIZE, 3])
    # img = tf.image.random_brightness(img, max_delta=32. / 255.)
    # img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    # img = tf.image.random_hue(img, max_delta=0.2)
    # img = tf.image.random_contrast(img, lower=0.5, upper=1.5)

    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    label = tf.cast(features['label'], tf.int32)
    pixes = tf.cast(features['pixes'], tf.int32)
    return img, label, pixes

def make_batch(filename,batch_size_in):
    img, label, pixes = decode_from_tfrecords(filename)
    min_after_deque = 1000
    batch_size = batch_size_in
    capacity = min_after_deque + 3 * batch_size

    image_batch, label_batch, pixes_batch = tf.train.shuffle_batch(
        [img, label, pixes], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_deque
    )

    return image_batch, label_batch, pixes_batch


def test_encode(path,filename):
    encode_to_tfrecords(path, filename)

def test_batch(filename):
    image_batch, label_batch, pixes_batch = make_batch(filename,50)
    coord = tf.train.Coordinator()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in range(2000):
            raw_image, raw_label, raw_pixes = sess.run([image_batch, label_batch, pixes_batch])
            print (raw_image, raw_label, raw_pixes)
    coord.request_stop()
    coord.join(threads)

def test_one_by_one(filename):
    image, label, pixes = decode_from_tfrecords(filename)
    coord = tf.train.Coordinator()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        i = 0
        for step in range(10):
            raw_image, raw_label, raw_pixes = sess.run([image, label, pixes])
            img = Image.fromarray(raw_image, 'RGB')

            img.save(SAME_SIZE_JPG+"_"+str(i)+".jpg")
            # print (raw_image, raw_label, raw_pixes)
            i = i + 1
    coord.request_stop()
    coord.join(threads)


def test_read_all(filename):
    image, label, pixel = decode_from_tfrecords(filename)
    coord = tf.train.Coordinator()
    init = tf.global_variables_initializer()
    images = []
    labels = []
    pixes = []
    with tf.Session() as sess:

        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        i = 0
        for step in range(10):
            raw_image, raw_label, raw_pixes = sess.run([image, label, pixel])
            images.append(raw_image)
            labels.append(raw_label)
            pixes.append(raw_pixes)
            print (raw_image, raw_label, raw_pixes)
            i = i + 1
    coord.request_stop()
    coord.join(threads)

    return images, labels, pixes




if __name__ == "__main__":
    # TRAIN VALIDATION DATA
    test_encode(PATH, TFRECORD_FILENAME)
    test_one_by_one(TFRECORD_FILENAME+".validation")

    # test_batch(TFRECORD_FILENAME)
    # print test_read_all(TFRECORD_FILENAME+".validation")


    # test_encode(TEST_PATH, TFRECORD_FILENAME)
    # test_one_by_one(TFRECORD_FILENAME + ".test")



