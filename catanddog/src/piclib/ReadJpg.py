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


BILINEAR_INTERPOLATION = 0
NEARST_NEIGHBOR_INTERPOLATION = 1
BICUBIC_INTERPOLATION = 2
AREA_INTERPOLATION = 3
class JPGReader(object):
    def __init__(self, srcfile, row_num ,col_num, method=0):
        self.__row_num = row_num
        self.__col_num = col_num
        self.__method = method
        self.__srcfile = srcfile
        self.__image_raw_data = tf.gfile.FastGFile(self.__srcfile, 'r').read()
        self.__img_data = tf.image.decode_jpeg(self.__image_raw_data)

    def set_file(self, srcfile):
        self.__srcfile = srcfile
        self.__image_raw_data = tf.gfile.FastGFile(self.__srcfile, 'r').read()
        self.__img_data = tf.image.decode_jpeg(self.__image_raw_data)

    def resize(self):
        # image_raw_data = tf.gfile.FastGFile(os.path.join(IMAGE_PATH, 'cat.1.jpg'), 'r').read()

        resized = tf.image.resize_images(self.__img_data, (self.__row_num, self.__col_num), method=self.__method)
        return resized

    def persistFile(self,destfile):
        encoded_image = tf.image.encode_jpeg(self.resize())
        with tf.gfile.GFile(destfile, "wb") as f:
            f.write(encoded_image.eval())
        return

    def distort_color(self, image, color_ordering=0):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32./255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32./255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        elif color_ordering == 2:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
        return tf.clip_by_value(image, 0.0, 1.0)

    def preprocess_for_train(self):
        return self.distort_color(tf.image.convert_image_dtype(resize, dtype=tf.float32), 0)



if __name__ == '__main__':
    IMAGE_SRC_PATH = "/home/zhangtx/ml/catanddog/data/train"
    IMAGE_DEST_PATH = "/home/zhangtx/ml/catanddog/data/dest"
    srcname = os.path.join(IMAGE_SRC_PATH, 'cat.0.jpg')
    destname = os.path.join(IMAGE_DEST_PATH, 'cat.0.jpg')
    jpgReader = JPGReader(srcname, 32, 32, method=NEARST_NEIGHBOR_INTERPOLATION)
    with tf.Session() as sess:
        resize = jpgReader.resize()
        plt.imshow(resize.eval())
        plt.show()
        jpgReader.persistFile(destname)
        distorted = sess.run(jpgReader.preprocess_for_train())
        plt.imshow(distorted)
        plt.show()

