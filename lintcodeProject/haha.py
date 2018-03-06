def sum(x, y):
    return x+y

p = lambda x, y: x + y

print p(1, 2)

key="add"
action={"reduce":lambda x,y:x-y, "add":lambda x,y:p(x,y)}
print action[key](2,4)

import tensorflow as tf
init = tf.global_variables_initializer();
with tf.Session() as sess:
    sess.run(init)
    labels = [1,3,5,7,9]
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, axis=1)
    # print sess.run(labels)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    # print sess.run(tf.range(0, batch_size, 1))
    concated = tf.concat([indices, labels],1)
    # print sess.run(concated)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 10]), 1.0, 0.0)
    print sess.run(onehot_labels)