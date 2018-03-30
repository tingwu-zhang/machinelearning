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
from PIL import Image , ImageFilter
from pylab import *

import ReadJpg

def __int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def __bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


NORMAL_PATH = "../../data/train/"

FLIP_LR_PATH = "../../data/flip_left_right_processed/"
FLIP_TB_PATH = "../../data/flip_top_bottom_processed/"
BLUR_PATH = "../../data/blur_processed/"
ROTATE_PATH = "../../data/rotate_processed/"
ROTATE90_PATH = "../../data/rotate90_processed/"


TFRECORD_LR_FILENAME = "../../data/dest/output.tfrecords.lr"
TFRECORD_NORMAL_FILENAME = "../../data/dest/output.tfrecords.normal"

TFRECORD_BLUR_FILENAME = "../../data/dest/output.tfrecords.blur"
TFRECORD_ROTATE_FILENAME = "../../data/dest/output.tfrecords.rotate"
TFRECORD_ROTATE90_FILENAME = "../../data/dest/output.tfrecords.rotate90"
TFRECORD_TB_FILENAME = "../../data/dest/output.tfrecords.tb"

SAME_SIZE_JPG = "../../data/dest/"

TEST_PATH = "../../data/test/"
RESIZED_IMAGE_SIZE = 64
TRAIN_BATCH_SIZE = RESIZED_IMAGE_SIZE

def pre_process(srcpath,destpath,method):

    for img_name in os.listdir(srcpath):

        img = Image.open(os.path.join(srcpath, img_name))

        if method == 0:
            final_image = img.rotate(45)
        elif method == 1:
            final_image = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif method == 2:
            final_image = img.filter(ImageFilter.BLUR)
        elif method == 3:
            final_image = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif method == 4:
            final_image = img.rotate(90)




        img_convert_ndarray = np.array(final_image)
        mean_img = Image.fromarray(img_convert_ndarray)
        mean_img.save(os.path.join(destpath, img_name))


        print os.path.join(srcpath, img_name) + ' is processed '



def encode_to_tfrecords(path, filename):
    writer_train_1 = tf.python_io.TFRecordWriter(filename+".train1")
    writer_train_2 = tf.python_io.TFRecordWriter(filename+".train2")
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
        print os.path.join(path, img_name) + ' is processed '

        chance = np.random.randint(1, 99)
        if chance < 12:
            writer_validation.write(example.SerializeToString())
            # writer_train.write(example.SerializeToString())
        else:
            if chance< 60:
                writer_train_1.write(example.SerializeToString())
            else:
                writer_train_2.write(example.SerializeToString())
    writer_train_1.close()
    writer_train_2.close()
    writer_validation.close()

def decode_from_tfrecords(file):
    filename_queue = tf.train.string_input_producer(file,  num_epochs=None)
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
    img = tf.image.resize_images(img, (TRAIN_BATCH_SIZE, TRAIN_BATCH_SIZE), method=tf.image.ResizeMethod.BILINEAR)
    img = tf.image.random_brightness(img, max_delta=32. / 255.)
    img = tf.image.per_image_standardization(img)
    # img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    # img = tf.image.random_hue(img, max_delta=0.2)
    # img = tf.image.random_contrast(img, lower=0.5, upper=1.5)

    # img = tf.image.convert_image_dtype(img, dtype=tf.float32, saturate=True)

    label = tf.cast(features['label'], tf.int32)
    pixes = tf.cast(features['pixes'], tf.int32)
    return img, label, pixes
    # return tf.multiply(tf.cast(img, tf.float32), 1.0/255.0), label, pixes

def make_batch(filename,batch_size_in):
    img, label, pixes = decode_from_tfrecords(filename)
    min_after_deque = 500
    batch_size = batch_size_in
    capacity = min_after_deque + 4 * batch_size

    image_batch, label_batch, pixes_batch = tf.train.shuffle_batch(
        [img, label, pixes], batch_size=batch_size, num_threads=6, allow_smaller_final_batch=False, capacity=capacity, min_after_dequeue=min_after_deque
    )
    # image_batch, label_batch, pixes_batch = tf.train.shuffle_batch_join([img, label, pixes],batch_size,capacity,min_after_deque)
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
        for step in range(1000):
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

            # img = Image.fromarray(raw_image, 'RGB')
            # img.save(SAME_SIZE_JPG+"_"+str(i)+".jpg")

            print (raw_image, raw_label, raw_pixes)

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
    # pre_process(NORMAL_PATH, ROTATE_PATH,0)
    # pre_process(NORMAL_PATH, FLIP_TB_PATH,1)
    # pre_process(NORMAL_PATH, BLUR_PATH,2)
    pre_process(NORMAL_PATH, FLIP_LR_PATH,3)
    pre_process(NORMAL_PATH, ROTATE90_PATH,4)

    # test_encode(NORMAL_PATH, TFRECORD_NORMAL_FILENAME)
    # test_encode(FLIP_TB_PATH, TFRECORD_TB_FILENAME)
    # test_encode(ROTATE_PATH, TFRECORD_ROTATE_FILENAME)
    # test_encode(BLUR_PATH, TFRECORD_BLUR_FILENAME)
    test_encode(FLIP_LR_PATH, TFRECORD_LR_FILENAME)
    test_encode(ROTATE90_PATH, TFRECORD_ROTATE90_FILENAME)
    # test_one_by_one(TFRECORD_FILENAME+".validation")

    # test_batch(TFRECORD_FILENAME)
    # print test_read_all(TFRECORD_FILENAME+".validation")


    # test_encode(TEST_PATH, TFRECORD_FILENAME)
    # test_one_by_one(TFRECORD_FILENAME + ".test")



