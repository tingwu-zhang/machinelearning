#!/usr/bin/env python

import tensorflow as tf
import os


def generate_tfrecords(input_filename, output_filename):
  print("Start to convert {} to {}".format(input_filename, output_filename))
  writer = tf.python_io.TFRecordWriter(output_filename)
  file = open(input_filename, "r")
  file.readline()
  for line in file:
      data = line.split(","),
      label = data[0]
      features = [i for i in data[1:783]]
      example = tf.train.Example(features=tf.train.Features(feature=
      {
          "label":tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
          "features":tf.train.Feature(int64_list=tf.train.Int64List(value=features))
      }))
      writer.write(example.SerializeToString())
  writer.close()
  print("Successfully convert {} to {}".format(input_filename,
                                               output_filename))




def print_tfrecords(input_filename):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([input_filename])
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
        'label': tf.FixedLenFeature([], tf.int64),
        'features': tf.FixedLenFeature([], tf.int64)})

    labels = tf.cast(features['label'], tf.int64)
    features = tf.cast(features['features'], tf.int64)
    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10):
        label, features = sess.run([labels, features])
        print label
        print features




def main():
    current_path = os.getcwd()
    for filename in os.listdir(current_path+"/data/"):
        if filename.startswith("") and filename.endswith(".csv"):
             generate_tfrecords(current_path+"/data/"+filename, current_path+"/data/"+filename + ".tfrecords")
             # print_tfrecords(current_path+"/data/"+filename + ".tfrecords")


if __name__ == "__main__":
  main()