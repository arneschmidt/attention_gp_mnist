import tensorflow as tf
import numpy as np

from data_generator import DataGenerator


class Data:
    def __init__(self, dataset):
        self.n_classes = 0
        self.dataset = dataset
        self.x_full, self.y_full, self.x_val, self.y_val, self.x_test, self.y_test = self.load_data()
        self.n_data_points = 60000
        # self.instances_in_bag_count = tf.zeros(shape=[10])
        self.instances_in_bag_count = [0,0,0,0,0,0,0,0,0,0]

    def load_data(self):
        if self.dataset == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            self.n_classes = 2
            (x_val, y_val) = (None, None)
        elif self.dataset == 'cifar10':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            self.n_classes = 3
            n_val = 10000
            (x_train, y_train) = (x_train[:(x_train.shape[0] - n_val)], y_train[:(x_train.shape[0] - n_val)])
            (x_val, y_val) = (x_train[(x_train.shape[0] - n_val):], y_train[(x_train.shape[0] - n_val):])
            x_val = x_val / 255
            y_val = self.n_classes - np.clip(y_val, a_min=0, a_max=self.n_classes - 1).astype(int) - 1

        x_full = x_train / 255
        y_full = self.n_classes - np.clip(y_train, a_min=0, a_max=self.n_classes-1).astype(int) - 1
        x_test = x_test / 255
        y_test = self.n_classes - np.clip(y_test, a_min=0, a_max=self.n_classes-1).astype(int) - 1

        return x_full, y_full, x_val, y_val, x_test, y_test

    def generate_train_data(self, bag_size=9, convert_to_bags=False):
        if self.dataset == 'mnist':
            train_gen = tf.data.Dataset.from_tensor_slices((self.x_full, self.y_full))
            train_gen = self._prepare_mnist_data(train_gen, bag_size, convert_to_bags=convert_to_bags)
        else:
            bag_x, bag_y = self._prepare_cifar10_data(split='train', convert_to_bags=convert_to_bags)
            train_gen = DataGenerator(bag_x, bag_y, shuffle=True)
            # train_gen = tf.data.Dataset.from_tensor_slices((bag_x, bag_y))

            for i in range(len(train_gen)):
                assert train_gen[i][0].shape == (9,32,32,3)
        return train_gen

    def generate_val_data(self, bag_size=9, convert_to_bags=False):
        if self.dataset == 'mnist':
            val_gen = None
        else:
            bag_x, bag_y = self._prepare_cifar10_data(split='test', convert_to_bags=convert_to_bags)
            val_gen = DataGenerator(bag_x, bag_y, shuffle=False)

        return val_gen

    def generate_test_data(self, bag_size=9, convert_to_bags=False):
        if self.dataset == 'mnist':
            test_gen = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
            test_gen = self._prepare_mnist_data(test_gen, bag_size, convert_to_bags=convert_to_bags)
        else:
            bag_x, bag_y = self._prepare_cifar10_data(split='test', convert_to_bags=convert_to_bags)
            test_gen = DataGenerator(bag_x, bag_y, shuffle=False)

        return test_gen

    def _prepare_mnist_data(self, ds, bag_size, convert_to_bags=True):

        def _convert_to_bags(x, y):
            y = tf.reduce_max(y, axis=0)
            y = tf.reshape(y, shape=[1])
            return x, y

        ds = ds.cache()
        # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
        ds = ds.batch(bag_size)
        if convert_to_bags:
            ds = ds.map(lambda x, y: _convert_to_bags(x, y))
        # ds = ds.map(lambda x, y: _reshape(x, y))
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def _prepare_cifar10_data(self, split='train', bag_size=9, convert_to_bags=True):
        if split == 'train':
            x = self.x_full
            y = self.y_full
        elif split == 'val':
            x = self.x_val
            y = self.y_val
        else:
            x = self.x_test
            y = self.y_test

        n_bags_per_class = np.floor(x.shape[0] / (bag_size * self.n_classes)).astype(np.int)
        self.n_data_points  = n_bags_per_class * bag_size * self.n_classes

        bag_y = []
        bag_inst_y = []
        bag_inst_x = []

        class_ids = []
        class_idx_start = []

        neg_class = 0

        for c in range(self.n_classes):
            class_ids.append(np.argwhere(y.flatten() == c).flatten()) # this index refers to the complete dataset (0:60000)
            class_idx_start.append(0)  # this index refers to the array of indices pers class (0:6000)

        class_idx_start = np.array(class_idx_start)
        for c in range(self.n_classes):
            for i in range(n_bags_per_class):
                n_negative = bag_size
                n_positive = 0

                # if positive bag, sample number of positive
                if c != neg_class:
                    n_positive = np.random.randint(1, 3)
                    n_negative = n_negative - n_positive
                    id_stop_pos = class_idx_start[c] + n_positive
                else: # in this case c == neg_class and we want to have an empty array
                    id_stop_pos = class_idx_start[c]

                id_stop_neg = class_idx_start[neg_class] + n_negative

                # reset counter in case it overshoots
                if id_stop_neg > class_ids[neg_class].shape[0]:
                    class_idx_start[neg_class] = 0
                    id_stop_neg = n_negative
                if id_stop_pos > class_ids[c].shape[0]:
                    class_idx_start[c] = 0
                    id_stop_pos = n_positive

                ids_neg = class_ids[neg_class][class_idx_start[neg_class]:id_stop_neg]
                ids_pos = class_ids[c][class_idx_start[c]:id_stop_pos]

                assert bag_size == ids_neg.shape[0] + ids_pos.shape[0]

                ids = np.concatenate([ids_neg, ids_pos], axis=0).flatten()

                bag_inst_x.append(x[ids])
                bag_inst_y.append(y[ids])
                bag_y.append(np.array([c]))

                class_idx_start[neg_class] += n_negative
                class_idx_start[c] += n_positive


        if convert_to_bags:
            bag_y = bag_y
        else:
            bag_y = bag_inst_y

        bag_inst_x = np.array(bag_inst_x)
        bag_y = np.array(bag_y)

        return bag_inst_x, bag_y










