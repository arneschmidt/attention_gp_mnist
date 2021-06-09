import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from data import Data
from model import build_model
from mil.data.datasets import mnist_bags
from evaluate import visualize_attention

def main():
    save_dir = './out_images/'
    os.makedirs(save_dir, exist_ok=True)
    bag_size = 9
    data_gen = Data()
    train_data = data_gen.generate_train_data(batch_size=bag_size)
    test_data_instances = data_gen.generate_test_data(batch_size=bag_size, convert_to_bags=False)
    test_data_bags = data_gen.generate_test_data(batch_size=bag_size, convert_to_bags=True)
    model, instance_model = build_model(data_dims=[28,28])

    model.summary()
    model.fit(train_data, batch_size=bag_size, epochs=5)
    print('Bag level evaluation:')
    model.evaluate(test_data_bags)

    visualize_attention(model, instance_model, test_data_instances, save_dir)

if __name__ == '__main__':
    main()