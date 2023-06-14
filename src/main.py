# TODO:
# visualize train/validation curve

import os
from data import Data
from model import build_model
from evaluate import visualize_attention, bag_level_evaluation, data_count, save_results, save_train_val_curve, get_bag_statistics
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.framework.ops import disable_eager_execution


def main():
    disable_eager_execution()
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    dataset = 'mnist' # 'mnist' or 'cifar10'
    bag_size = 9
    epochs = 5
    save_dir = 'out_gp'
    attention = ['gp'] #['gp', 'bnn_gauss', 'bnn_mcdrop', 'mean_agg', 'att_det', 'att_det_gated']
    n_repetitions = 1
    save_train_hist = True

    save_dir = save_dir + '_' + dataset
    os.makedirs(save_dir, exist_ok=True)

    results = pd.DataFrame()
    dataset_stats = pd.Series()

    for j in range(len(attention)):
        for i in range(n_repetitions):
            print('####### Model: ' + attention[j] + '  Run: ' + str(i) + ' #######')
            np.random.seed(i)
            tf.random.set_seed(i)

            data_gen = Data(dataset)
            train_data = data_gen.generate_train_data(bag_size=bag_size, convert_to_bags=True)
            val_data = data_gen.generate_val_data(bag_size=bag_size, convert_to_bags=True)
            test_data_bags = data_gen.generate_test_data(bag_size=bag_size, convert_to_bags=True)

            train_data_instances = data_gen.generate_train_data(bag_size=bag_size, convert_to_bags=False)
            test_data_instances = data_gen.generate_test_data(bag_size=bag_size, convert_to_bags=False)
            get_bag_statistics(train_data_instances)
            model, instance_model, bag_level_uncertainty_model = build_model(attention=attention[j], dataset=dataset)

            model.summary()
            # hist = model.fit(train_data, epochs=1, steps_per_epoch=1)
            hist = model.fit(train_data, validation_data=val_data, epochs=epochs)

            print('Bag level evaluation:')
            results = bag_level_evaluation(test_data_bags, bag_level_uncertainty_model, dataset=dataset, model=attention[j], run=i, results=results)
            # if attention[j] != 'att_det' and attention[j] != 'att_det_gated':
            #     visualize_attention(bag_level_uncertainty_model, instance_model, test_data_instances, save_dir, dataset, save_images=False)

            if attention[j] =='gp' and i == 0 and save_train_hist and dataset=='cifar10':
                save_train_val_curve(save_dir, hist)
            tf.keras.backend.clear_session()

    dataset_stats = data_count(train_data_instances, mode='train', dataset=dataset, dataset_stats=dataset_stats)
    dataset_stats = data_count(test_data_instances, mode='test', dataset=dataset, dataset_stats=dataset_stats)

    save_results(save_dir, results, dataset_stats)


if __name__ == '__main__':
    main()