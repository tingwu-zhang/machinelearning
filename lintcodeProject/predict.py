# -*- coding: utf-8 -*-
import time
import tensorflow as tf
import inference
import lintorc
import numpy as np


BATCH_SIZE = 600
LEARNING_RATE_BASE = 0.5
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99


MODEL_SAVE_PATH = "/home/zhangtx/ml/lintcodeProject/model"
MODEL_NAME = "model.ckpt"

DEFAULT_LABEL = 0
DEFAULT_FEATURE = 0.0

FILENAME = "./data/test.csv"
SUBMISSION = "./data/submission.csv"
import csv
import numpy as np

# from tensorflow.examples.tutorials.mnist import input_data



def oneShot(curr_y_train_batch):
    num_labels = curr_y_train_batch.shape[0]
    index_offset = np.arange(num_labels) * 10
    num_labels_hot = np.zeros((num_labels, 10))
    num_labels_hot.flat[index_offset+curr_y_train_batch.ravel()] = 1.0
    return num_labels_hot

def evalate():
    with tf.Graph().as_default() as g:

        x = tf.placeholder(tf.float32, [None, inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name='y-input')

        sess = tf.Session()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        y = inference.inference(x, None)
        prediction = tf.argmax(y,1)

        variable_average = tf.train.ExponentialMovingAverage(lintorc.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        coord = tf.train.Coordinator()

        test_data = np.loadtxt(open(FILENAME), delimiter=",", skiprows=1)

        tmpXs = test_data * 1.0/255.0

        xs = np.matrix(tmpXs)

        curr_x_train_batch = xs.astype(float)


        # validata_feed = {x: mnist.validation.images,
        #                  y_: mnist.validation.labels}
        #
        predict_feed = {x: curr_x_train_batch}

        ckpt = tf.train.get_checkpoint_state(lintorc.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            predictvalue = sess.run(prediction, feed_dict=predict_feed)
            with open(SUBMISSION, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["ImageId", "Label"])
                for i in range(28000):
                    writer.writerow([i+1, predictvalue[i]])

            print predictvalue.shape
        sess.close()




def main(argv=None):
    evalate()
if __name__ == '__main__':
    tf.app.run()


