# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 40000
MOVING_AVERAGE_DECAY = 0.99
FEATURE_NUM = 10

MODEL_SAVE_PATH = "/home/zhangtx/ml/pm25predict/model"
MODEL_NAME = "model.ckpt"

DEFAULT_FEATURE = [[''], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

FILENAME = "../data/train/train.csv"
TESTFILENAME = "../data/test.csv"

def read_data(file_queue):

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    defaults = DEFAULT_FEATURE
    train_item = tf.decode_csv(value, defaults)
    value = train_item[2:3]

    feature = train_item[3:13]
    # pdb.set_trace()
    # feature = []
    # for i in range(FEATURE_NUM-3):
    #     feature.append(train_item[3+i])

    return feature, value

def create_pipeline(filename, batch_size, num_epochs=None):

    file_queue = tf.train.string_input_producer([filename], shuffle=True, num_epochs=num_epochs)
    example, label = read_data(file_queue)

    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * batch_size

    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue, num_threads=1
    )

    return example_batch, label_batch


def train():
    # W = tf.Variable(tf.random_uniform([10, 1], -1.0, 4.0, dtype=tf.float32), name="weight")
    W = tf.Variable(tf.truncated_normal([FEATURE_NUM, 1], mean=0.0, stddev=0.1, dtype=tf.float32,name="weight"))
    b = tf.Variable(tf.zeros([1, 1]), name="b", dtype=tf.float32)

    x = tf.placeholder(tf.float32, [None, FEATURE_NUM], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, 1], name="y-input")

    y = tf.add(tf.matmul(x, W), b)

    loss = tf.reduce_mean(tf.square(y - y_), name="loss")
    tf.summary.scalar("loss", loss)
    train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    # train = tf.train.AdamOptimizer(0.01).minimize(loss)


    xs, ys = create_pipeline(FILENAME, BATCH_SIZE, num_epochs=2000)


    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("../logs/", tf.get_default_graph())


    merged_summary = tf.summary.merge_all()



    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        count = 0

        while not coord.should_stop():

            xs_batch, ys_batch = sess.run([xs, ys])

            # 计算相关系
            # txs_batch = np.reshape(xs_batch, (FEATURE_NUM, BATCH_SIZE))
            # conv = np.corrcoef(txs_batch)
            # print conv
            #
            # z-score stadardlization
            # txs_batch = np.reshape(xs_batch, (FEATURE_NUM, BATCH_SIZE))
            # txs_batch = np.reshape(xs_batch, (FEATURE_NUM, BATCH_SIZE))
            t_txs_batch = np.transpose(xs_batch)
            conv = np.corrcoef(t_txs_batch)
            # print conv
            for i in range(FEATURE_NUM):
                txs_batch_mean = np.mean(t_txs_batch[i])
                txs_batch_std = np.std(t_txs_batch[i])

                if txs_batch_mean != 0 and txs_batch_std != 0:
                    t_txs_batch[i] = (t_txs_batch[i]-txs_batch_mean)/txs_batch_std


            xs_batch = np.transpose(t_txs_batch)

            ys_batch = ys_batch * 0.01
            for i in range(BATCH_SIZE):
                xs_batch[i][0] = xs_batch[i][0]   # dew_point
                xs_batch[i][1] = xs_batch[i][1]    # temperature
                xs_batch[i][2] = xs_batch[i][2]   #pressure
                xs_batch[i][3] = xs_batch[i][3]      #wind_speed
                xs_batch[i][4] = xs_batch[i][4]   # snow_time
                xs_batch[i][5] = xs_batch[i][5]   # rain_time
                xs_batch[i][6] = xs_batch[i][6]    #wind_ne
                xs_batch[i][7] = xs_batch[i][7]    #wind_nw
                xs_batch[i][8] = xs_batch[i][8]   #wind_se
                xs_batch[i][9] = xs_batch[i][9]      #wind_cv
            #
            #
            feed_dict = {x: xs_batch, y_: ys_batch}
            _, loss_value, summary = sess.run([train, loss, merged_summary],
                                                    feed_dict=feed_dict

                                             )
            if count % 100 == 0:
                print loss_value
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=count)
            count = count + 1
            writer.add_summary(summary, count)


    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    coord.join(threads)
    sess.close()

def main(argv=None):
    train()
if __name__ == '__main__':
    tf.app.run()


