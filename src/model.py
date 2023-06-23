import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import numpy as np
from layers import Mil_Attention


class RBFKernelFn(tf.keras.layers.Layer):
    """
    RBF kernel for Gaussian processes.
    """
    def __init__(self, trainable, lengthscale, variance, **kwargs):
        super(RBFKernelFn, self).__init__(**kwargs)
        dtype = kwargs.get('dtype', None)
        self.trainable = trainable
        self.lengthscale = lengthscale
        self.variance = variance

        self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(2.0),
            dtype=dtype,
            name='amplitude')

        self._length_scale = self.add_variable(
            initializer=tf.constant_initializer(-1.0),
            dtype=dtype,
            name='length_scale')

    def call(self, x):
        # Never called -- this is just a layer so it can hold variables
        # in a way Keras understands.
        return x

    @property
    def kernel(self):
        # tf.print('amp', tf.nn.softplus(0.1 * self._amplitude))
        # tf.print('ls', tf.nn.softplus(10.0 * self._length_scale))
        if self.trainable:
            amp = tf.nn.softplus(self._amplitude)
            ls = tf.nn.softplus(self._length_scale)
        else:
            amp = self.variance
            ls = self.lengthscale

        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude= amp, # tf.nn.softplus(self._amplitude), #
            length_scale= ls, # tf.nn.softplus(self._length_scale) #
        )


def build_model(attention, config, data_dims=[28,28]):
    num_training_points = 60000
    num_classes = 2
    batch_size = 8
    mc_samples = 20

    def mc_sampling(x):
        """
        Monte Carlo Sampling of the GP output distribution.
        :param x:
        :return:
        """
        samples = x.sample(mc_samples)
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

    def custom_softmax(x):
        x = tf.reshape(x, shape=[mc_samples, 1, -1])
        # x = tf.reshape(x, shape=[1, -1])
        x = tf.keras.activations.softmax(x, axis=-1)
        out = tf.reshape(x, shape=[mc_samples, -1])
        # out = tf.reshape(x, shape=[-1])
        return out

    def attention_multiplication(i):
        # a = tf.ones_like(i[0])
        a = i[0]
        f = i[1]
        # tf.print('attention', a)
        # tf.print('features', f)
        if attention == 'gp':
            out = tf.linalg.matvec(f, a, transpose_a=True)
        else:
            a = tf.reshape(a, shape=[-1])
            out = tf.linalg.matvec(f, a, transpose_a=True)
            out = tf.reshape(out, [1, f.shape[1]])
        return out

    def reshape_final(x):
        out = tf.reshape(x, shape=[num_classes])
        return out

    input = tf.keras.layers.Input(shape=data_dims)
    # x = tf.keras.layers.Flatten()(input)
    # x = tf.keras.layers.Dense(128, activation='relu')(x)
    # # x = tf.keras.layers.Dense(64, activation='relu')(x)
    # f = tf.keras.layers.Dense(8, activation=None)(x)
    x = tf.reshape(input, shape=(-1, 28,28,1))
    x = tf.keras.layers.Conv2D(4, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(-1, 28, 28, 1))(x)
    x = tf.keras.layers.MaxPool2D()(x)
    # x = tf.keras.layers.Conv2D(4, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(-1, 28, 28, 1))(input)
    # x = tf.keras.layers.MaxPool2D()(x)
    # x = tf.keras.layers.Dense(64, activation='relu')
    x = tf.keras.layers.Flatten()(x)
    f = tf.keras.layers.Dense(config['cnn_feature_dims'], activation='relu')(x)
    if attention == 'gp':
        if config['att_activation'] == 'sigmoid':
            x = tf.keras.layers.Dense(config['feature_dims'], activation='sigmoid')(f)
        elif config['att_activation'] == 'linear':
            x = tf.keras.layers.Dense(config['feature_dims'], activation=None)(f)
        else:
            x = f
        x = tfp.layers.VariationalGaussianProcess(
            mean_fn=lambda x: tf.ones([1]) * config['prior_mean'],
            num_inducing_points=config['num_inducing_points'],
            kernel_provider=RBFKernelFn(config['kernel_trainable'], config['kernel_lengthscale'],
                                        config['kernel_variance']),
            event_shape=[1],  # output dimensions
            inducing_index_points_initializer=tf.keras.initializers.RandomUniform(
                minval=config['indp_min'], maxval=config['indp_max'], seed=None
            ),
            jitter=10e-5,
            convert_to_tensor_fn=tfp.distributions.Distribution.sample,
            unconstrained_observation_noise_variance_initializer=tf.initializers.constant(config['obsnoise_var_init']),
            variational_inducing_observations_scale_initializer=tf.initializers.constant(
                config['indp_var_init'] * np.tile(np.eye(config['num_inducing_points'], config['num_inducing_points']), (1, 1, 1))),
            )(x)

        x = tf.keras.layers.Lambda(mc_sampling, name='instance_attention')(x)
        # x = tf.reshape(x, [-1, 20])
        # x = tf.keras.layers.Activation('sigmoid')(x)
        # x = tf.keras.layers.Activation('softmax')(x)
        a = tf.keras.layers.Lambda(custom_softmax, name='instance_softmax')(x)
        x = tf.keras.layers.Lambda(attention_multiplication)([a,f])

        x = tf.reshape(x, shape=[mc_samples, 1, -1])
        # x = tf.reshape(x, shape=[ 1, -1])
        # x = tf.keras.layers.Dense(8, activation='relu')(x)
        x = tf.keras.layers.Dense(num_classes, activation='softmax',  name='bag_softmax')(x)
        output = tf.keras.layers.Lambda(mc_integration)(x)
    else:
        a = Mil_Attention(f.shape[1], output_dim=0, name='instance_softmax', use_gated=False)(f)
        x = tf.keras.layers.Lambda(attention_multiplication)([a, f])
        # x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(num_classes, activation='softmax', name='bag_softmax')(x)
        output = tf.keras.layers.Lambda(reshape_final)(x)

    # output = tf.reshape(x, shape=[1])
    # output = tf.expand_dims(x, axis=1)

    model = tf.keras.Model(inputs=input, outputs=output, name="sgp_mil")
    if attention == 'gp':
        model.add_loss(kl_loss(model, batch_size, num_training_points, config))
    # model.build()

    instance_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer('instance_attention').output)
    bag_level_uncertainty_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer('bag_softmax').output)

    optimizers = [
        tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        tf.keras.optimizers.Adam(learning_rate=config['learning_rate_gp'])
    ]
    optimizers_and_layers = [(optimizers[0], model.layers[:7]), (optimizers[1], model.layers[7]),
                             (optimizers[0], model.layers[8:])]

    optimizer = tfa.optimizers.discriminative_layer_training.MultiOptimizer(optimizers_and_layers)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model, instance_model, bag_level_uncertainty_model

def kl_loss(head, batch_size, num_training_points, config):
    # tf.print('kl_div: ', kl_div)
    num_training_points = tf.constant(num_training_points, dtype=tf.float32)
    batch_size = tf.constant(batch_size, dtype=tf.float32)

    layer_name = 'variational_gaussian_process'
    vgp_layer = head.get_layer(layer_name)

    def _kl_loss():
        # kl_weight = tf.cast(0.001 * batch_size / num_training_points, tf.float32)
        kl_weight = tf.cast(1.0 / num_training_points, tf.float32)*config['kl_weight']
        kl_div = tf.reduce_sum(vgp_layer.submodules[5].surrogate_posterior_kl_divergence_prior())

        loss = tf.multiply(kl_weight, kl_div)
        # tf.print('kl_weight: ', kl_weight)
        # tf.print('kl_loss: ', loss)
        # # tf.print('u_var: ', head.variables[4])
        return loss

    return _kl_loss