# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf
import numpy as np
import csv
BATCH_SIZE = 144

FEATURE_NUM = 11

MODEL_SAVE_PATH = "/home/zhangtx/ml/pm25predict/model"
MODEL_NAME = "model.ckpt"

DEFAULT_FEATURE = [[''], [0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]


TESTFILENAME = "../data/train/validation.csv"
SUBMISSION = "../data/train/submission.csv"


def predict(feature,result):
    with tf.Graph().as_default() as g:
        # W = tf.Variable(tf.truncated_normal([FEATURE_NUM, 1], 0.0, 1.0, dtype=tf.float32, name="weight"))
        W = tf.Variable(tf.truncated_normal([FEATURE_NUM, 1], mean=0.0, stddev=0.1, dtype=tf.float32, name="weight"))

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
        features = tf.stack(train_item[2:13])

        day = (train_item[0:1])
        hour = (train_item[1:2])
        real = ((train_item[2:3]))
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
                    write_real = sess.run(real)
                    matrix_all.append(xs)
                    day_all.append(write_day)
                    hour_all.append(write_hour)
                    real_all.append(write_real)

                txs_batch = np.transpose(matrix_all)

                for i in range(FEATURE_NUM):
                    txs_batch_mean = np.mean(txs_batch[i])
                    txs_batch_std = np.std(txs_batch[i])

                    if txs_batch_mean != 0 and txs_batch_std != 0:
                        txs_batch[i] = (txs_batch[i] - txs_batch_mean) / txs_batch_std
                xs_batch = np.transpose(txs_batch)

                # for i in range(BATCH_SIZE):
                #     xs_batch[i][0] = xs_batch[i][0]   # dew_point
                #     xs_batch[i][1] = xs_batch[i][1]   # temperature
                #     xs_batch[i][2] = xs_batch[i][2]   # pressure
                #     xs_batch[i][3] = xs_batch[i][3]   # wind_speed
                #     xs_batch[i][4] = xs_batch[i][4]   # snow_time
                #     xs_batch[i][5] = xs_batch[i][5]   # rain_time
                #     xs_batch[i][6] = xs_batch[i][6]   # wind_ne
                #     xs_batch[i][7] = xs_batch[i][7]   # wind_nw
                #     xs_batch[i][8] = xs_batch[i][8]   # wind_se
                #     xs_batch[i][9] = xs_batch[i][9]   # wind_cv

                xs_batch = np.reshape(xs_batch, (BATCH_SIZE, FEATURE_NUM))
                validata_feed = {x: xs_batch}
                count = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                prediction_value = sess.run(prediction, feed_dict=validata_feed)

                sum = 0
                for i in range(BATCH_SIZE):
                    writer.writerow((day_all[i][0], hour_all[i][0], np.abs(prediction_value[i][0]) * 100, real_all[i][0]))
                    sum = sum + (prediction_value[i][0] * 100-real_all[i][0])*(np.abs(prediction_value[i][0]) * 100-real_all[i][0])
                print("step %s ,Mean square error is %e" % (count,sum/BATCH_SIZE))


                coord.request_stop()
                coord.join(threads)


def main(argv=None):
    while True:
        predict(TESTFILENAME, SUBMISSION)
        time.sleep(10)



if __name__ == '__main__':
    tf.app.run()


