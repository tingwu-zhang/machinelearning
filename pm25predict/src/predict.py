# -*- coding: utf-8 -*-
import os

import tensorflow as tf
import numpy as np
import csv
BATCH_SIZE = 5989

MOVING_AVERAGE_DECAY = 0.99
FEATURE_NUM = 10

MODEL_SAVE_PATH = "/home/zhangtx/ml/pm25predict/model"
MODEL_NAME = "model.ckpt"

DEFAULT_FEATURE = [[''], [0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
# DEFAULT_FEATURE = [[''], [0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]


TESTFILENAME = "../data/test/test.csv"
SUBMISSION = "../data/train/submission.csv"


def predict(feature,result):
    with tf.Graph().as_default() as g:
        W = tf.Variable(tf.truncated_normal([FEATURE_NUM, 1], 0.0, 1.0, dtype=tf.float32, name="weight"))
        b = tf.Variable(tf.zeros([1, 1]), name="b", dtype=tf.float32)

        x = tf.placeholder(tf.float32, [None, FEATURE_NUM], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, 1], name="y-input")

        prediction = tf.add(tf.matmul(x, W), b)

        saver = tf.train.Saver()

        init = tf.global_variables_initializer()


        # pdb.set_trace()

        filename_queue = tf.train.string_input_producer([feature])

        reader = tf.TextLineReader(skip_header_lines=1)
        key, value = reader.read(filename_queue)

        train_item = tf.decode_csv(
            value, record_defaults=DEFAULT_FEATURE)
        features = tf.stack(train_item[2:12])

        day = (train_item[0:1])
        hour = (train_item[1:2])
        # real = ((train_item[2:3]))
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            saver.restore(sess, ckpt.model_checkpoint_path)
            with open(result, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["date", "hour", "pm2.5"])

                matrix_all = []
                day_all = []
                hour_all = []
                real_all = []
                for i in range(BATCH_SIZE):
                    xs = sess.run(features)
                    write_day = sess.run(day)
                    write_hour = sess.run(hour)
                    # write_real = sess.run(real)
                    matrix_all.append(xs)
                    day_all.append(write_day)
                    hour_all.append(write_hour)
                    # real_all.append(write_real)

                txs_batch = np.transpose(matrix_all)
                for i in range(FEATURE_NUM):
                    txs_batch_mean = np.mean(txs_batch[i])
                    txs_batch_std = np.std(txs_batch[i])
                    if txs_batch_mean != 0 and txs_batch_std != 0:
                        txs_batch[i] = (txs_batch[i] - txs_batch_mean) / txs_batch_std
                xs_batch = np.transpose(txs_batch)
                for i in range(BATCH_SIZE):
                    feature_x = xs_batch
                    # feature_x = feature_x * 0.1
                    feature_x[0] = feature_x[0]   # dew_point
                    feature_x[1] = feature_x[1]   # temperature
                    feature_x[2] = feature_x[2]   # pressure
                    feature_x[3] = feature_x[3]   # wind_speed
                    feature_x[4] = feature_x[4]   # snow_time
                    feature_x[5] = feature_x[5]   # rain_time
                    feature_x[6] = feature_x[6]   # wind_ne
                    feature_x[7] = feature_x[7]   # wind_nw
                    feature_x[8] = feature_x[8]   # wind_se
                    feature_x[9] = feature_x[9]   # wind_cv


                xs_batch = np.reshape(txs_batch, (BATCH_SIZE, FEATURE_NUM))
                validata_feed = {x: xs_batch}
                count = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                prediction_value = sess.run(prediction, feed_dict=validata_feed)

                sum = 0
                for i in range(BATCH_SIZE):
                    writer.writerow((day_all[i][0], hour_all[i][0], prediction_value[i][0] * 100))
                    # sum = sum + (prediction_value[i][0] * 100-real_all[i][0])*(prediction_value[i][0] * 100-real_all[i][0])
                # print("Mean square error is %d" % (sum/BATCH_SIZE))


                coord.request_stop()
                coord.join(threads)


def main(argv=None):
    predict(TESTFILENAME, SUBMISSION)


if __name__ == '__main__':
    tf.app.run()


