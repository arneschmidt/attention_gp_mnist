import os
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE


def visualize_attention(bag_level_uncertainty_model, instance_model, test_gen_instances, save_dir, quick_eval=True):
    n_visualize = 10
    test_data = test_gen_instances.as_numpy_iterator()

    n = len(test_gen_instances)
    correct_attention_preds = 0
    correct_attention_std = 0
    wrong_attention_preds = 0
    wrong_attention_std = 0

    pos_bags = 0
    for i in range(n):
        x_bag, y_bag = next(test_data)
        bag_pred = bag_level_uncertainty_model.predict(x_bag)
        bag_mean = np.mean(bag_pred, axis=0)[0][1]
        bag_std = np.std(bag_pred, axis=0)[0][1]

        preds = instance_model.predict(x_bag)
        mean = np.reshape(np.mean(preds, axis=0), [-1])
        std = np.reshape(np.std(preds, axis=0), [-1])
        correct_attention, att_std = _correct_att_prediction(mean, std, y_bag)
        if correct_attention:
            correct_attention_preds += 1
            correct_attention_std += att_std
        else:
            wrong_attention_preds += 1
            wrong_attention_std += att_std
        # pos_attention = np.reshape((mean > 0.0), [-1])
        # correct_attention = pos_attention == y_bag
        # wrong_attention = np.logical_not(correct_attention)
        # correct_attention_preds += np.sum(correct_attention)
        # wrong_attention_preds += np.sum(wrong_attention)
        # correct_attention_std += np.sum(std[correct_attention])
        # wrong_attention_std += np.sum(std[wrong_attention])


        pos_bags += np.max(y_bag)
        if i < n_visualize or int(np.max(y_bag)) != int(np.round(bag_mean)):
            _plot_images(x_bag, bag_mean, bag_std, mean, std, os.path.join(save_dir + str(i)) + '.jpg')
        else:
            if quick_eval:
                break
    attention_accuracy = correct_attention_preds/(correct_attention_preds + wrong_attention_preds)
    correct_att_std = correct_attention_std/correct_attention_preds
    wrong_att_std = wrong_attention_std/wrong_attention_preds

    mlflow.log_metric('attention_accuracy', attention_accuracy)
    mlflow.log_metric('att_std_correct', correct_att_std)
    mlflow.log_metric('att_std_wrong', wrong_att_std)

    print('Attention accuracy: ',  str(attention_accuracy))
    print('Correct attention std: ',  str(correct_att_std))
    print('Wrong attention std: ',  str(wrong_att_std))

def _plot_images(x_bag, bag_mean, bag_std, mean, std, save_path):
    f, axarr = plt.subplots(3, 3)
    f.suptitle('Bag prediction: ' + str(np.round(bag_mean, 2)) + '±' + str(np.round(bag_std, 2)), fontsize=16)
    for i in range(3):
        for j in range(3):
            n = (i*3) + j
            axarr[i, j].imshow(x_bag[n], cmap='gray_r')
            axarr[i, j].set_title("Att. " + str(np.round(mean[n], 2)) + '±' + str(np.round(std[n], 2)) )
    f.tight_layout(pad=1.5)
    plt.savefig(save_path)
    plt.close()

def _correct_att_prediction(mean, std, instance_gt):
    am = np.argmax(mean)
    am_std = std[am]
    if instance_gt[am] == 1:
        return 1, am_std
    else:
        return 0, am_std

def bag_level_evaluation(test_gen, bag_level_uncertainty_model):
    test_data = test_gen.as_numpy_iterator()

    n = len(test_gen)
    correct_preds_count = 0
    correct_stds_count = 0
    wrong_preds_count = 0
    wrong_stds_count = 0

    for i in range(n):
        x_bag, y_bag = next(test_data)
        bag_pred = bag_level_uncertainty_model.predict(x_bag)
        mean = np.mean(bag_pred, axis=0)[0][1]
        std = np.std(bag_pred, axis=0)[0][1]
        correct_pred = (np.round(mean) == y_bag)[0]
        if correct_pred:
            correct_preds_count += 1
            correct_stds_count += std
        else:
            wrong_preds_count += 1
            wrong_stds_count += std

        # if i == 0:
        #     stds = std
        #     correct_preds = correct_pred
        # else:
        #     stds = np.concatenate([stds, std])
        #     correct_preds = np.concatenate([correct_preds, correct_pred])

    accuracy = correct_preds_count/n
    correct_pred_std = correct_stds_count/correct_preds_count
    wrong_pred_std = wrong_stds_count/wrong_preds_count

    mlflow.log_metric('bag_accuracy', accuracy)
    mlflow.log_metric('bag_std_correct', correct_pred_std)
    mlflow.log_metric('bag_std_wrong', wrong_pred_std)


    print('Bag Accuracy: ' + str(accuracy)
          + '; correct_pred_std: ' + str(correct_pred_std)
          + '; wrong_pred_std: ' + str(wrong_pred_std))
def print_tsne_evaluation(instance_model, train_data_instances, save_name_tsne, save_name_att, config):
    feature_extractor = tf.keras.models.Sequential(instance_model.layers[:7])

    test_data = train_data_instances.as_numpy_iterator()

    inducing_point_locs = instance_model.trainable_variables[7].numpy()[0]
    inducing_point_stds = np.diag(instance_model.trainable_variables[9][0].numpy())
    inducing_point_means = instance_model.trainable_variables[8][0].numpy()

    n = 50
    features = []
    means = []
    stds = []
    labels = []
    for i in range(n):
        x = next(test_data)
        bag_features = feature_extractor.predict(x)
        bag_preds = instance_model.predict(x)
        features.append(bag_features)
        mean = np.reshape(np.mean(bag_preds, axis=0), [-1])
        std = np.reshape(np.std(bag_preds, axis=0), [-1])
        means.append(mean)
        stds.append(std)
        labels.append(x[1])


    noise_factor = 0.6
    for i in range(n):
        x = next(test_data)
        img = x[0]

        sampled_pixel_noise_ids = np.random.choice([False, True], size=img.shape,
                                                   p=[1 - noise_factor, noise_factor])
        img[sampled_pixel_noise_ids] = 1.0
        bag_features = feature_extractor.predict(img)
        bag_preds = instance_model.predict(img)
        features.append(bag_features)
        mean = np.reshape(np.mean(bag_preds, axis=0), [-1])
        std = np.reshape(np.std(bag_preds, axis=0), [-1])
        means.append(mean)
        stds.append(std)
        labels.append(x[1])

    noise_factor = 0.8
    for i in range(n):
        x = next(test_data)
        img = x[0]

        sampled_pixel_noise_ids = np.random.choice([False, True], size=img.shape,
                                                   p=[1 - noise_factor, noise_factor])
        img[sampled_pixel_noise_ids] = 1.0
        bag_features = feature_extractor.predict(img)
        bag_preds = instance_model.predict(img)
        features.append(bag_features)
        mean = np.reshape(np.mean(bag_preds, axis=0), [-1])
        std = np.reshape(np.std(bag_preds, axis=0), [-1])
        means.append(mean)
        stds.append(std)
        labels.append(x[1])

    features = np.array(features).reshape([-1, config['feature_dims']])
    means = np.array(means).reshape([-1])
    stds = np.array(stds).reshape([-1])
    labels = np.array(labels).reshape([-1])

    feat_indpoints = np.concatenate([inducing_point_locs, features], axis=0)
    stds = np.concatenate([inducing_point_stds, stds])
    means = np.concatenate([inducing_point_means, means])
    labels = np.concatenate([np.zeros_like(inducing_point_stds), labels])

    if feat_indpoints.shape[1] > 2:
        print('Calculating TSNE')
        feat_2d = TSNE(n_components=2, n_iter=1000, random_state=0, n_jobs=4).fit_transform(feat_indpoints)
    else:
        feat_2d = feat_indpoints

    indices_ind_points = np.arange(0,config['num_inducing_points'])
    indices_no_noise = np.arange(config['num_inducing_points'], n*9+config['num_inducing_points'])
    indices_low_noise= np.arange(n*9+config['num_inducing_points'], 2*n*9+config['num_inducing_points'])
    indices_high_noise= np.arange(2*n*9+config['num_inducing_points'], 3*n*9+config['num_inducing_points'])
    indices_negative = indices_no_noise[labels[indices_no_noise] == 0]
    indices_positive = indices_no_noise[labels[indices_no_noise] == 1]

    fig = plt.figure()
    plot = fig.add_subplot(1, 1, 1)
    plot.scatter(feat_2d[indices_no_noise,1], feat_2d[indices_no_noise,0], c=np.clip(stds[indices_no_noise], a_min=0.0, a_max=0.2),  vmin=0., vmax=0.2, marker='o', edgecolors='k', cmap='Greens', label='No noise')
    plot.scatter(feat_2d[indices_low_noise,1], feat_2d[indices_low_noise,0], c=np.clip(stds[indices_low_noise], a_min=0.0, a_max=0.2), vmin=0., vmax=0.2, marker='s', edgecolors='k', cmap='Greens', label='Low noise')
    plot.scatter(feat_2d[indices_high_noise,1], feat_2d[indices_high_noise,0], c=np.clip(stds[indices_high_noise], a_min=0.0, a_max=0.2), vmin=0., vmax=0.2, marker='p', edgecolors='k', cmap='Greens', label='High noise')
    plot.scatter(feat_2d[indices_ind_points,1], feat_2d[indices_ind_points,0], c=np.clip(stds[indices_ind_points], a_min=0.0, a_max=0.2), vmin=0., vmax=0.2, marker='^', edgecolors='k', cmap='Reds', label='Inducing points')
    plot.legend()
    fig.savefig(os.path.join(config['exp_dir'], save_name_tsne))

    fig = plt.figure()
    plot = fig.add_subplot(1, 1, 1)
    plot.scatter(means[indices_low_noise], stds[indices_low_noise], marker='s', edgecolors='k', c='tab:olive', label='Low Noise')
    plot.scatter(means[indices_high_noise], stds[indices_high_noise], marker='p', edgecolors='k', c='tab:green', label='High Noise')
    plot.scatter(means[indices_negative], stds[indices_negative], marker='o', edgecolors='k', c='tab:blue', label='Negative')
    plot.scatter(means[indices_positive], stds[indices_positive], marker='o', edgecolors='k', c='tab:purple', label='Positive')
    plot.scatter(means[indices_ind_points], stds[indices_ind_points], marker='', edgecolors='k', c='k', label='Inducing Points')
    plot.legend()
    fig.savefig(os.path.join(config['exp_dir'], save_name_att))


    mlflow.log_metrics({'std_inducing_points': float(np.mean(stds[indices_ind_points])),
                        'std_low_noise': float(np.mean(stds[indices_low_noise])),
                        'std_high_noise': float(np.mean(stds[indices_high_noise])),
                        'std_negative': float(np.mean(stds[indices_negative])),
                        'std_positive': float(np.mean(stds[indices_positive])),
                        'std_diff_pos_ood': float(np.mean(stds[indices_positive])) - float(np.mean(stds[indices_high_noise])),
                        'mean_inducing_points': float(np.mean(means[indices_ind_points])),
                        'mean_low_noise': float(np.mean(means[indices_low_noise])),
                        'mean_high_noise': float(np.mean(means[indices_high_noise])),
                        'mean_negative': float(np.mean(means[indices_negative])),
                        'mean_positive': float(np.mean(means[indices_positive])),
                        'mean_diff_pos_ood': float(np.mean(means[indices_positive])) - float(np.mean(means[indices_high_noise])),
                        })




    print('STD no noise: ' + str(np.mean(stds[indices_no_noise])))
    print('STD low noise: ' + str(np.mean(stds[indices_low_noise])))
    print('STD high noise: ' + str(np.mean(stds[indices_high_noise])))
    print('STD inducing points: ' + str(np.mean(stds[indices_ind_points])))









