import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from layers import Mil_Attention


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
        # tf.print('amp', tf.nn.softplus(0.1 * self._amplitude))
        # tf.print('ls', tf.nn.softplus(10.0 * self._length_scale))
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=tf.nn.softplus(0.1 * self._amplitude), # 0.1
            length_scale=tf.nn.softplus(10.0 * self._length_scale) # 5.
        )


def build_model(attention, dataset):
    num_inducing_points = 64
    num_training_points = 8000
    batch_size = 8
    inst_bag_dim = 8
    feature_dim = 8
    mc_samples = 20
    bag_size = 9

    if dataset == 'mnist':
        data_dims = [28, 28]
        num_classes = 2
    elif dataset == 'cifar10':
        data_dims = [32, 32, 3]
        num_classes = 3
    else:
        print('Choose valid dataset!')

    def mc_sampling(x):
        """
        Monte Carlo Sampling of the GP output distribution.
        :param x:
        :return:
        """
        samples = x.sample(mc_samples)
        return samples

    def mc_dropout_sampling(x):
        x_concat = tf.keras.layers.Dropout(0.5,  name='instance_attention')(x, training=True)
        x_concat = tf.expand_dims(x_concat, axis=0)
        for i in range(mc_samples-1):
            x_i = tf.keras.layers.Dropout(0.5,  name='instance_attention')(x, training=True)
            x_i = tf.expand_dims(x_i, axis=0)
            x_concat = tf.concat([x_concat, x_i], axis=0)
        return x_concat

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
        if attention != 'att_det' and attention != 'att_det_gated' and attention != 'mean_agg':
            out = tf.linalg.matvec(f, a, transpose_a=True)
        else:
            a = tf.reshape(a, shape=[-1])
            out = tf.linalg.matvec(f, a, transpose_a=True)
            out = tf.reshape(out, [1, f.shape[1]])
        return out

    def reshape_final(x):
        out = tf.reshape(x, shape=[num_classes])
        return out

    def att_mean(x):
        dims = x.shape[1]
        x = tf.reduce_mean(x, axis=0)
        out = tf.reshape(x, [1, dims])
        return out

    input = tf.keras.layers.Input(shape=data_dims)
    if dataset == 'mnist':
        x = tf.reshape(input, shape=(-1, 28, 28, 1))
        x = tf.keras.layers.Conv2D(4, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(-1, 28, 28, 1))(x)
        x = tf.keras.layers.MaxPool2D()(x)
    elif dataset == 'cifar10':
        # x = tf.reshape(input, shape=(bag_size, 32, 32, 3))
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    f = tf.keras.layers.Dense(64, activation='relu')(x)
    if attention == 'gp':
        x = tf.keras.layers.Dense(32, activation='sigmoid')(f)
        x = tfp.layers.VariationalGaussianProcess(
            mean_fn=lambda x: tf.ones([1]) * 0.0,
            num_inducing_points=num_inducing_points,
            kernel_provider=RBFKernelFn(),
            event_shape=[1],  # output dimensions
            inducing_index_points_initializer=tf.keras.initializers.RandomUniform(
                minval=0.3, maxval=0.7, seed=None
            ),
            jitter=10e-3,
            convert_to_tensor_fn=tfp.distributions.Distribution.sample,
            variational_inducing_observations_scale_initializer=tf.initializers.constant(
                0.01 * np.tile(np.eye(num_inducing_points, num_inducing_points), (1, 1, 1))),
            )(x)

        x = tf.keras.layers.Lambda(mc_sampling, name='instance_attention')(x)
        a = tf.keras.layers.Lambda(custom_softmax, name='instance_softmax')(x)
        x = tf.keras.layers.Lambda(attention_multiplication)([a, f]) # dim: (20,9,1)

        x = tf.reshape(x, shape=[mc_samples, 1, -1]) # dim: (20,4)
        x = tf.keras.layers.Dense(num_classes, activation='softmax',  name='bag_softmax')(x) # (20,1,64)
        output = tf.keras.layers.Lambda(mc_integration)(x)  # (20,1,3)
    elif attention == 'bnn_mcdrop':
        x = tf.keras.layers.Dense(32, activation='relu')(f)
        x = tf.keras.layers.Lambda(mc_dropout_sampling, name='instance_attention')(x)
        x = tf.keras.layers.Dense(1, activation='relu')(x)
        a = tf.keras.layers.Lambda(custom_softmax, name='instance_softmax')(x)
        x = tf.keras.layers.Lambda(attention_multiplication)([a, f])

        x = tf.reshape(x, shape=[mc_samples, 1, -1])
        x = tf.keras.layers.Dense(num_classes, activation='softmax',  name='bag_softmax')(x)
        output = tf.keras.layers.Lambda(mc_integration)(x)
    elif attention == 'bnn_gauss':
        x = tfp.layers.DenseReparameterization(32, kernel_posterior_tensor_fn=lambda d: d.sample(mc_samples))(f)
        # x = tfp.layers.DenseReparameterization(1, kernel_posterior_tensor_fn=lambda d: d.sample(mc_samples))(x)
        x = tf.keras.layers.Dense(1, activation='relu')(x)
        a = tf.keras.layers.Lambda(custom_softmax, name='instance_softmax')(x) # dim: (20, 9, 1)
        x = tf.keras.layers.Lambda(attention_multiplication)([a, f])

        x = tf.reshape(x, shape=[mc_samples, 1, -1])
        x = tf.keras.layers.Dense(num_classes, activation='softmax',  name='bag_softmax')(x)
        output = tf.keras.layers.Lambda(mc_integration)(x)
    elif attention=='att_det':
        a = Mil_Attention(f.shape[1], output_dim=0, name='instance_softmax', use_gated=False)(f)
        x = tf.keras.layers.Lambda(attention_multiplication)([a, f])
        # x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(num_classes, activation='softmax', name='bag_softmax')(x)
        output = tf.keras.layers.Lambda(reshape_final)(x)
    elif attention == 'att_det_gated':
        a = Mil_Attention(f.shape[1], output_dim=0, name='instance_softmax', use_gated=True)(f)
        x = tf.keras.layers.Lambda(attention_multiplication)([a, f])
        # x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(num_classes, activation='softmax', name='bag_softmax')(x)
        output = tf.keras.layers.Lambda(reshape_final)(x)
    elif attention=='mean_agg':
        x = tf.keras.layers.Lambda(att_mean, name='instance_softmax')(x)
        x = tf.keras.layers.Dense(num_classes, activation='softmax', name='bag_softmax')(x)
        output = tf.keras.layers.Lambda(reshape_final)(x)

    model = tf.keras.Model(inputs=input, outputs=output, name="mil_model")
    if attention == 'gp':
        model.add_loss(kl_loss(model, batch_size, num_training_points))

    instance_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer('instance_softmax').output)
    bag_level_uncertainty_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer('bag_softmax').output)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model, instance_model, bag_level_uncertainty_model


def kl_loss(head, batch_size, num_training_points):
    # tf.print('kl_div: ', kl_div)
    num_training_points = tf.constant(num_training_points, dtype=tf.float32)
    batch_size = tf.constant(batch_size, dtype=tf.float32)

    layer_name = 'variational_gaussian_process'
    vgp_layer = head.get_layer(layer_name)
    # layer_no = 13
    # vgp_layer = head.layers[layer_no]

    def _kl_loss():
        kl_weight = tf.cast(1.0 / num_training_points, tf.float32)
        kl_div = tf.reduce_sum(vgp_layer.submodules[5].surrogate_posterior_kl_divergence_prior())

        loss = tf.multiply(kl_weight, kl_div)
        return loss

    return _kl_loss