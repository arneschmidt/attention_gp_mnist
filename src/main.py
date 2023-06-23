import os
import argparse
import yaml
import mlflow
from data import Data
from model import build_model
from evaluate import print_tsne_evaluation, visualize_attention, bag_level_evaluation
import tensorflow as tf

def main(config):
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)
    bag_size = 9
    epochs = config['epochs']
    save_dir = config['exp_dir']
    attention = 'gp' #'gp' or 'deterministic'
    # os.makedirs(save_dir, exist_ok=True)

    mlflow.set_tracking_uri(config["mlflow_url"])
    experiment_id = mlflow.set_experiment(experiment_name='attention_gp_mnist')
    mlflow.start_run(experiment_id=experiment_id, run_name=save_dir.split('/')[-2])
    mlflow.log_params(config)

    data_gen = Data()
    train_data_instances = data_gen.generate_train_data(batch_size=bag_size, convert_to_bags=False)
    train_data = data_gen.generate_train_data(batch_size=bag_size, convert_to_bags=True)
    test_data_instances = data_gen.generate_test_data(batch_size=bag_size, convert_to_bags=False)
    test_data_bags = data_gen.generate_test_data(batch_size=bag_size, convert_to_bags=True)
    model, instance_model, bag_level_uncertainty_model = build_model(attention=attention, config=config,
                                                                     data_dims=[28,28])

    model.summary()
    print_tsne_evaluation(instance_model, train_data_instances, 'TSNE_before_train.png', 'stat_before_train', config=config)
    # model.fit(train_data, batch_size=bag_size, epochs=1, steps_per_epoch=1)
    model.fit(train_data, batch_size=bag_size, epochs=epochs)
        # visualize_attention(model, instance_model, test_data_instances, save_dir, quick_eval=True)
    print('Bag level evaluation:')
    # model.evaluate(test_data_bags)
    if attention == 'gp':
        print_tsne_evaluation(instance_model, train_data_instances, 'TSNE_after_train.png', 'stat_after_train', config=config)
        mlflow.log_artifacts(save_dir)
        bag_level_evaluation(test_data_bags, bag_level_uncertainty_model)
        visualize_attention(bag_level_uncertainty_model, instance_model, test_data_instances, save_dir, quick_eval=False)
    else:
        model.evaluate(test_data_bags)
    mlflow.log_artifacts(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cancer Classification")
    parser.add_argument("--experiment_folder", "-e", type=str, default="None",
                        help="Config path to experiment config. Parameters will override defaults. Optional.")
    args = parser.parse_args()
    if args.experiment_folder == 'None':
        experiment_folder = './exp/default/'
    else:
        experiment_folder = args.experiment_folder

    with open(os.path.join(experiment_folder, 'config.yaml')) as file:
        config = yaml.full_load(file)
    config['exp_dir'] = experiment_folder
    main(config)
