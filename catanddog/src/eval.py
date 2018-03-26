# -*- coding: utf-8 -*-
import time
import tensorflow as tf
import inferency
import piclib.Transpose
import numpy as np


VALIDATION_NUM = 1000
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
FILENAME = "../data/dest/output.tfrecords.validation"

import numpy as np

# from tensorflow.examples.tutorials.mnist import input_data



def oneShot(curr_y_train_batch):
    num_labels = curr_y_train_batch.shape[0]
    index_offset = np.arange(num_labels) * 2
    num_labels_hot = np.zeros((num_labels, 2))
    num_labels_hot.flat[index_offset+curr_y_train_batch.ravel()] = 1.0
    return num_labels_hot

def evalate(filename):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [VALIDATION_NUM,
                                        inferency.IMAGE_SIZE,
                                        inferency.IMAGE_SIZE,
                                        inferency.NUM_CHANNELS], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, inferency.OUTPUT_NODE], name='y-input')

        y = inferency.interfence(x, False, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        image, label, pixel = piclib.Transpose.decode_from_tfrecords(filename)
        coord = tf.train.Coordinator()
        images = []
        labels = []
        pixes = []
        with tf.Session() as sess:
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            i = 0
            for step in range(VALIDATION_NUM):
                raw_image, raw_label, raw_pixes = sess.run([image, label, pixel])
                images.append(raw_image)
                labels.append(raw_label)
                pixes.append(raw_pixes)
            coord.request_stop()
            coord.join(threads)
            labelsarray = np.asarray(labels)
            imgesarray = np.asarray(images)
            reshaped_ys = oneShot(labelsarray)

            reshaped_xs = np.reshape(imgesarray, (VALIDATION_NUM,
                                                  inferency.IMAGE_SIZE,
                                                  inferency.IMAGE_SIZE,
                                                  inferency.NUM_CHANNELS))

            validata_feed = {x: reshaped_xs, y_: reshaped_ys}
            while True:
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validata_feed)
                    print("After %s training steps,validation accuracy =%g" % (global_step, accuracy_score))
                time.sleep(EVAL_INTERVAL_SEC)


def main(argv=None):
    evalate(FILENAME)


if __name__ == '__main__':
    tf.app.run()


