import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from data import generate_data
from model import build_model


def main():
    x, y, y_bags = generate_data()
    model, instance_model = build_model(2)

    model.summary()
    ds = tf.data.Dataset.from_tensor_slices((x, y_bags))
    ds = ds.shuffle(buffer_size=1000)
    model.fit(ds, batch_size=3, epochs=20)
    for i in range(100):
        preds = instance_model.predict_generator(x[i])
        print('instance_p', preds)
        print('instance_g', y[i])


if __name__ == '__main__':
    main()