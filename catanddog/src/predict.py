# -*- coding: utf-8 -*-
import time
import tensorflow as tf
import inferency
import piclib.Transpose
import numpy as np
import csv
from PIL import Image
from pylab import *


TEST_NUM = 4999
LEARNING_RATE_BASE = 0.5
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99


MODEL_SAVE_PATH = "../model"
MODEL_NAME = "model.ckpt"

DEFAULT_LABEL = 0
DEFAULT_FEATURE = 0.0
EVAL_INTERVAL_SEC = 10
FILENAME = "../data/dest/output.tfrecords.test"
SUBMISSION = "../data/dest/submission.csv"

import numpy as np
import os
# from tensorflow.examples.tutorials.mnist import input_data

TEST_PATH = "../data/test/"

def oneShot(curr_y_train_batch):
    num_labels = curr_y_train_batch.shape[0]
    index_offset = np.arange(num_labels) * 2
    num_labels_hot = np.zeros((num_labels, 2))
    num_labels_hot.flat[index_offset+curr_y_train_batch.ravel()] = 1.0
    return num_labels_hot

def predict(path,filename):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [1,
                                        inferency.IMAGE_SIZE,
                                        inferency.IMAGE_SIZE,
                                        inferency.NUM_CHANNELS], name="x-input")
        y_ = tf.placeholder(tf.float32, [1, inferency.OUTPUT_NODE], name='y-input')

        y = inferency.interfence(x, False, None)
        prediction = tf.argmax(y,1)

        variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        image, label, pixes = piclib.Transpose.decode_from_tfrecords(filename)
        coord = tf.train.Coordinator()
        init = tf.global_variables_initializer()


        with tf.Session() as sess:

            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            saver.restore(sess, ckpt.model_checkpoint_path)
            with open(SUBMISSION, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["id", "label"])

                for i in range(5000):
                    img_name = os.path.join(path, str(i)+".jpg")
                    print img_name
                    img = Image.open(img_name)
                    img = img.resize((piclib.Transpose.RESIZED_IMAGE_SIZE, piclib.Transpose.RESIZED_IMAGE_SIZE))
                    img = np.multiply(img,1.0/100.0)
                    reshaped_xs = np.reshape(img, (1,
                                                     inferency.IMAGE_SIZE,
                                                     inferency.IMAGE_SIZE,
                                                     inferency.NUM_CHANNELS))
                    validata_feed = {x: reshaped_xs}
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    prediction_value = sess.run(prediction, feed_dict=validata_feed)

                    writer.writerow([i + 1, prediction_value[0]])

def main(argv=None):
    predict(TEST_PATH, FILENAME)


if __name__ == '__main__':
    tf.app.run()


