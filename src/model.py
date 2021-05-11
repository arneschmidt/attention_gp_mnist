import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class RBFKernelFn(tf.keras.layers.Layer):
    """
    RBF kernel for Gaussian processes.
    """
    def __init__(self, **kwargs):
        super(RBFKernelFn, self).__init__(**kwargs)
        dtype = kwargs.get('dtype', None)

        self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(0.0),
            dtype=dtype,
            name='amplitude')

        self._length_scale = self.add_variable(
            initializer=tf.constant_initializer(0.0),
            dtype=dtype,
            name='length_scale')

    def call(self, x):
        # Never called -- this is just a layer so it can hold variables
        # in a way Keras understands.
        return x

    @property
    def kernel(self):
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=tf.nn.softplus(0.1 * self._amplitude), # 0.1
            length_scale=tf.nn.softplus(10.0 * self._length_scale) # 5.
        )


def build_model(data_dims=2):
    num_inducing_points = 10
    num_training_points = 8000
    num_classes = 1
    batch_size = 8
    inst_bag_dim = 8

    def mc_sampling(x):
        """
        Monte Carlo Sampling of the GP output distribution.
        :param x:
        :return:
        """
        samples = x.sample(20)
        return samples


    def mc_integration(x):
        """
        Monte Carlo integration is basically replacing an integral with the mean of samples.
        Here we take the mean of the previously generated samples.
        :param x:
        :return:
        """
        x = tf.math.reduce_mean(x, axis=0)
        out = tf.reshape(x, [-1])
        return out

    def attention_multiplication(i):
        a = i[0]
        f = i[1]
        # tf.print('attention', a)
        # tf.print('features', f)

        out = tf.linalg.matvec(f, a, transpose_a=True)
        return out

    input = tf.keras.layers.Input(shape=[data_dims])
    f = tf.keras.layers.Dense(8, activation=None)(input)
    x = tf.keras.layers.Activation('sigmoid')(f)
    x = tfp.layers.VariationalGaussianProcess(
        # mean_fn=tf.constant(-5),
        num_inducing_points=num_inducing_points,
        kernel_provider=RBFKernelFn(),
        event_shape=[1],  # output dimensions
        inducing_index_points_initializer=tf.keras.initializers.RandomUniform(
            minval=0.0, maxval=1.0, seed=None
        ),
        jitter=10e-3,
        convert_to_tensor_fn=tfp.distributions.Distribution.sample,
        variational_inducing_observations_scale_initializer=tf.initializers.constant(
            0.01 * np.tile(np.eye(num_inducing_points, num_inducing_points), (num_classes, 1, 1))),
        )(x)

    x = tf.keras.layers.Lambda(mc_sampling)(x)
    # x = tf.reshape(x, [-1, 20])
    x = tf.keras.layers.Activation('sigmoid')(x)
    a = tf.keras.layers.Lambda(mc_integration)(x)
    # a = tf.keras.layers.Activation('softmax')(a)
    x = tf.keras.layers.Lambda(attention_multiplication)([a,f])
    x = tf.reshape(x, shape=[1, -1])
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    output = tf.reshape(x, shape=[1])
    # output = tf.expand_dims(x, axis=1)

    model = tf.keras.Model(inputs=input, outputs=output, name="sgp_mil")
    model.add_loss(kl_loss(model, batch_size, num_training_points))
    # model.build()

    model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.1),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model

def kl_loss(head, batch_size, num_training_points):
    # tf.print('kl_div: ', kl_div)
    num_training_points = tf.constant(num_training_points, dtype=tf.float32)
    batch_size = tf.constant(batch_size, dtype=tf.float32)

    def _kl_loss():
        # kl_weight = tf.cast(0.001 * batch_size / num_training_points, tf.float32)
        kl_weight = tf.cast(0.001 * batch_size / num_training_points, tf.float32)
        kl_div = tf.reduce_sum(head.layers[3].submodules[5].surrogate_posterior_kl_divergence_prior())

        loss = tf.multiply(kl_weight, kl_div)
        # tf.print('kl_weight: ', kl_weight)
        # tf.print('kl_loss: ', loss)
        # # tf.print('u_var: ', head.variables[4])
        return loss

    return _kl_loss