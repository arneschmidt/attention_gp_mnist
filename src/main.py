import os
from data import Data
from model import build_model
from evaluate import visualize_attention, bag_level_evaluation
import tensorflow as tf

def main():
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)
    bag_size = 9
    epochs = 5
    save_dir = 'out_images_deterministic/'
    attention = 'gp' #'gp' or 'deterministic'
    os.makedirs(save_dir, exist_ok=True)

    data_gen = Data()
    train_data = data_gen.generate_train_data(batch_size=bag_size)
    test_data_instances = data_gen.generate_test_data(batch_size=bag_size, convert_to_bags=False)
    test_data_bags = data_gen.generate_test_data(batch_size=bag_size, convert_to_bags=True)
    model, instance_model, bag_level_uncertainty_model = build_model(attention=attention, data_dims=[28,28])

    model.summary()
    # model.fit(train_data, batch_size=bag_size, epochs=1, steps_per_epoch=1)
    model.fit(train_data, batch_size=bag_size, epochs=epochs)
        # visualize_attention(model, instance_model, test_data_instances, save_dir, quick_eval=True)
    print('Bag level evaluation:')
    # model.evaluate(test_data_bags)
    if attention == 'gp':
        bag_level_evaluation(test_data_bags, bag_level_uncertainty_model)
        visualize_attention(bag_level_uncertainty_model, instance_model, test_data_instances, save_dir, quick_eval=False)
    else:
        model.evaluate(test_data_bags)


if __name__ == '__main__':
    main()