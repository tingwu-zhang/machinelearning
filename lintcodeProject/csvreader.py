import tensorflow as tf

FEATURE_COUNT = 784
DEFAULT_LABEL=0
DEFAULT_FEATURE=0.0


filename_queue = tf.train.string_input_producer(["./data/train.csv"])

reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.

record_defaults = [[0]]
for i in range(FEATURE_COUNT):
    record_defaults.append([0.0])

train_item = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack(train_item[1:])

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1):
    # Retrieve a single instance:
    example, label = sess.run([features, train_item[0]])
    print example.shape, label.shape


  coord.request_stop()
  coord.join(threads)