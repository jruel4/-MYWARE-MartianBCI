

import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.constant(0, dtype=tf.float32)

update = tf.assign(W, W*2)

linear_model = W * x + b

init = tf.global_variables_initializer()


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#print(sess.run(init))

print(sess.run([W,b,x]) )















