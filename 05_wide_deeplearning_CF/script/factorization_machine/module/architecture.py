import tensorflow as tf


class FM(tf.keras.Model):

    def __init__(self, n_feature, n_latent_feature):
        super(FM, self).__init__()

        self.w_0 = tf.Variable([0.0])
        self.w = tf.Variable(tf.zeros([n_feature]))
        self.V = tf.Variable(tf.random.normal(shape=(n_feature, n_latent_feature)))

    def call(self, inputs):

        linear_terms = tf.reduce_sum(tf.math.multiply(self.w, inputs), axis=1)

        interactions = 0.5 * tf.reduce_sum(input_tensor=tf.math.pow(tf.matmul(inputs, self.V), 2) - tf.matmul(tf.math.pow(inputs, 2), tf.math.pow(self.V, 2)),
                                           axis=1,
                                           keepdims=False)

        y_hat = tf.math.sigmoid(self.w_0 + linear_terms + interactions)
        
        return y_hat