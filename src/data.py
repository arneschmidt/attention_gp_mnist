import tensorflow as tf
import numpy as np

class Data:
    def __init__(self):
        self.x_full, self.y_full, self.x_test, self.y_test = self.load_mnist_data()
        self.n_data_points = 60000

    def load_mnist_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_full = x_train / 255
        y_full = 1.0 - np.clip(y_train, a_min=0, a_max=1).astype(int)
        x_test = x_test / 255
        y_test = 1.0 - np.clip(y_test, a_min=0, a_max=1).astype(int)
        return x_full, y_full, x_test, y_test

    def generate_train_data(self, batch_size=128, convert_to_bags=False):
        train_gen = tf.data.Dataset.from_tensor_slices((self.x_full, self.y_full))
        train_gen = self._prepare_data(train_gen, batch_size, convert_to_bags=convert_to_bags)
        return train_gen

    def generate_test_data(self, batch_size=128, convert_to_bags=False):
        test_gen = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        test_gen = self._prepare_data(test_gen, batch_size, convert_to_bags=convert_to_bags)
        return test_gen

    def _prepare_data(self, ds, batch_size, convert_to_bags=True):
        def _convert_to_bags(x, y):
            y = tf.reduce_max(y, axis=0)
            y = tf.reshape(y, shape=[1])
            return x, y
        #
        # def _reshape(x, y):
        #     x = tf.reshape(x, shape=[batch_size, 28,28,1])
        #     return x, y

        ds = ds.cache()
        # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
        ds = ds.batch(batch_size)
        if convert_to_bags:
            ds = ds.map(lambda x, y: _convert_to_bags(x, y))
        # ds = ds.map(lambda x, y: _reshape(x, y))
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

