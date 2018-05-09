import tensorflow as tf

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def dense(x, n_in, n_out, scope=None, with_w=False, xavier = False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [n_in, n_out], tf.float32, tf.contrib.layers.xavier_initializer() if xavier else tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [n_out], initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias
