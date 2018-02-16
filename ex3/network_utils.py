import tensorflow as tf


def unpool(value, name='unpool'):
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


def build_classification_network(model, keep_prob):
    model = tf.contrib.layers.flatten(model)
    model = tf.layers.dense(model, 576, activation=tf.nn.relu)
    model = tf.nn.dropout(model, keep_prob)
    model = tf.layers.dense(model, 384, activation=tf.nn.relu)
    model = tf.nn.dropout(model, keep_prob)
    model = tf.layers.dense(model, 192, activation=tf.nn.relu)
    model = tf.nn.dropout(model, keep_prob)
    return tf.add(tf.matmul(model, tf.Variable(tf.truncated_normal([model.shape[1].value, 10], stddev=0.1))),
                  tf.Variable(tf.zeros([10])))


def print_stats(epoch, session, inputs, feature_batch, targets, label_batch,
                valid_features, valid_labels, keep_prob, cost, accuracy):
    feed_cost = {inputs: feature_batch, targets: label_batch, keep_prob: 1.0}
    feed_valid = {inputs: valid_features, targets: valid_labels, keep_prob: 1.0}
    cost = session.run(cost, feed_cost)
    accuracy = session.run(accuracy, feed_valid)
    print('Epoch {:>2}:  '.format(epoch + 1))
    print("cost: %.2f" % cost, "accuracy: %.2f" % accuracy)
    pass
