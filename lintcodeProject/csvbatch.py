# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import inference



BATCH_SIZE = 5
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

RECORD_NUM = 42000

MODEL_SAVE_PATH = "/home/zhangtx/ml/lintcodeProject/model"
MODEL_NAME = "model.ckpt"

DEFAULT_LABEL = 0
DEFAULT_FEATURE = 0.0
FILENAME = "./data/val.csv"
TESTFILENAME = "./data/test.csv"
#读取函数定义
def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    reader.num_records_produced()
    #定义列
    print key
    record_defaults = [[DEFAULT_LABEL]]
    for i in range(inference.INPUT_NODE):
        record_defaults.append([DEFAULT_FEATURE])

    #编码
    train_item = tf.decode_csv(value, record_defaults)

    feature = train_item[1:]
    label = train_item[0:1]
    return tf.stack(feature), tf.stack(label)

def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], shuffle=False, num_epochs=num_epochs)
    example, label = read_data(file_queue)
    min_after_dequeue = 20
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )
    return example_batch, label_batch

x_train_batch, y_train_batch = create_pipeline(FILENAME, BATCH_SIZE, num_epochs=2)




global_step = tf.Variable(0, trainable=False)
learning_rate = 0.1


init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

sess = tf.Session()

sess.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    print("Training: ")
    count = 0

    while not coord.should_stop():
        curr_x_train_batch = sess.run(x_train_batch)
        curr_y_train_batch = sess.run(y_train_batch)

        train_batch = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        for i  in range(BATCH_SIZE-1):
            train_batch.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


        for i in range(BATCH_SIZE):
            train_batch[i][curr_y_train_batch[i][0]] = 1
            print curr_x_train_batch[i], train_batch[i], curr_y_train_batch[i][0]

        # sess.run(train_step, feed_dict={
        #     x: curr_x_train_batch,
        #     y: train_batch
        # })

        count += 1
        # ce, summary = sess.run([cross_entropy, merged_summary], feed_dict={
        #     x: curr_x_train_batch,
        #     y: train_batch
        # })
        print("After %s training steps,validation accuracy =%g" % (count, 0))

        # train_writer.add_summary(summary, count)

        # ce, test_acc, test_summary = sess.run([cross_entropy, accuracy, merged_summary], feed_dict={
        #     x: curr_x_test_batch,
        #     y: curr_y_test_batch
        # })
        # test_writer.add_summary(summary, count)
        # print('Batch', count, 'J = ', ce, '测试准确率=', test_acc)
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()