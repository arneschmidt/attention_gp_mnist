import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import sem

import tensorflow_datasets as tfds


def data_count(generator, mode, dataset, dataset_stats):
    n = len(generator)
    if dataset == 'mnist':
        generator = generator.as_numpy_iterator()

    if dataset == 'mnist':
        n_classes = 2
    else:
        n_classes = 3

    instance_count = []
    bag_count = []

    for c in range(n_classes):
        instance_count.append(0)
        bag_count.append(0)

    for i in range(n):
        if dataset == 'mnist':
            x_bag, y_bag = next(generator)
        else:
            x_bag, y_bag = generator[i]
        y_inst = y_bag.flatten()
        y_bag = np.max(y_inst)
        for c in range(n_classes):
            instance_count[c] += int(np.sum(y_inst == c))
            bag_count[c] += int(y_bag == c)

    for c in range(n_classes):
        dataset_stats[mode + '_inst_count_' + str(c)] = instance_count[c]
        dataset_stats[mode + '_bag_count_' + str(c)] = bag_count[c]

    return dataset_stats


def visualize_attention(bag_level_uncertainty_model, instance_model, test_gen_instances, save_dir, dataset, save_images=True):
    if dataset == 'mnist':
        test_data = test_gen_instances.as_numpy_iterator()
    else:
        test_data = test_gen_instances
    save_dir = os.path.join(save_dir + 'images')
    n_visualize = 100

    n = len(test_gen_instances)
    correct_attention_preds = 0
    correct_attention_std = 0
    wrong_attention_preds = 0
    wrong_attention_std = 0

    pos_bags = 0


    for i in range(n):
        if dataset == 'mnist':
            x_bag, y_bag = next(test_data)
        else:
            x_bag, y_bag = test_data[i]
        bag_pred = bag_level_uncertainty_model.predict(x_bag)
        bag_mean = np.mean(bag_pred, axis=0)[0][1]
        bag_std = np.std(bag_pred, axis=0)[0][1]

        preds = instance_model.predict(x_bag)
        mean = np.reshape(np.mean(preds, axis=0), [-1])
        std = np.reshape(np.std(preds, axis=0), [-1])

        bag_class = np.max(y_bag)
        if bag_class > 0:
            correct_attention, att_std = _correct_att_prediction(mean, std, y_bag)
            if correct_attention:
                correct_attention_preds += 1
                correct_attention_std += att_std
            else:
                wrong_attention_preds += 1
                wrong_attention_std += att_std
            pos_bags += 1

        if save_images and (i < n_visualize or int(bag_class) != int(np.round(bag_mean))):
            _plot_images(x_bag, bag_mean, bag_std, mean, std, os.path.join(save_dir + str(i)) + '.jpg')

    attention_accuracy = correct_attention_preds/(correct_attention_preds + wrong_attention_preds + 0.000001)
    correct_att_std = correct_attention_std/(correct_attention_preds + 0.000001)
    wrong_att_std = wrong_attention_std/(wrong_attention_preds + 0.000001)

    # results['Attention_accuracy'] = attention_accuracy
    # results['Correct_attention_std'] = correct_att_std
    # results['Wrong_attention_std'] = wrong_att_std

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
    if instance_gt[am] != 0:
        return 1, am_std
    else:
        return 0, am_std

def bag_level_evaluation(test_gen, bag_level_uncertainty_model, dataset, model, run, results):
    if dataset == 'mnist':
        test_data = test_gen.as_numpy_iterator()
    else:
        test_data = test_gen

    n = len(test_gen)

    class_pred = []
    gt = []
    correct_stds = []
    wrong_stds = []

    for i in range(n):
        if model == 'att_det' or model == 'att_det_gated' or model == 'mean_agg':
            if dataset == 'mnist':
                x_bag, y_bag = next(test_data)
            else:
                x_bag, y_bag = test_data[i]
            bag_pred = bag_level_uncertainty_model.predict(x_bag)
            mean = np.argmax(bag_pred[0], axis=0)
            correct_stds.append(0)
            wrong_stds.append(0)
        else:
            if dataset == 'mnist':
                x_bag, y_bag = next(test_data)
                bag_pred = bag_level_uncertainty_model.predict(x_bag)
                mean = np.round(np.mean(bag_pred, axis=0)[0][1])
                std = np.std(bag_pred, axis=0)[0][1]
                correct_pred = (np.round(mean) == y_bag)[0]
            else:
                x_bag, y_bag = test_data[i]
                bag_pred = bag_level_uncertainty_model.predict(x_bag)
                mean = np.argmax(np.mean(bag_pred, axis=0)[0])
                std = np.sum(np.std(bag_pred, axis=0)[0])
                correct_pred = (mean == y_bag)[0]
            if correct_pred:
                correct_stds.append(std)
            else:
                wrong_stds.append(std)
        class_pred.append(mean)
        gt.append(y_bag)

    class_pred = np.array(class_pred)
    gt = np.array(gt)
    
    acc = accuracy_score(class_pred, gt)
    f1score = f1_score(class_pred, gt, average='macro')
    correct_std = np.mean(correct_stds)
    wrong_std = np.mean(wrong_stds)

    row = pd.DataFrame({'model': model, 'run': run, 'acc': acc, 'f1score': f1score, 'correct_std': correct_std,
                        'wrong_std': wrong_std}, index=[0])

    results = pd.concat([results, row], ignore_index=True)

    print('Bag Accuracy: ' + str(acc) + '; F1score: ' + str(f1score)
          + '; correct_pred_std: ' + str(correct_std)
          + '; wrong_pred_std: ' + str(wrong_std))
    return results

def save_results(save_dir, results, dataset_stats):
    out_path = os.path.join(save_dir, 'results.csv')
    results.to_csv(out_path)
    dataset_stats.to_csv(os.path.join(save_dir, 'dataset_stats.csv'))

    models = np.unique(results['model'])

    df_results = pd.DataFrame()
    for model in models:
        model_df = results[results['model']==model]
        dict = {'model': model,
                'mean_acc': np.mean(model_df['acc']),
                'se_acc': sem(model_df['acc']),
                'mean_f1score': np.mean(model_df['f1score']),
                'se_f1score': sem(model_df['f1score']),
                'correct_std': np.mean(model_df['correct_std']),
                'wrong_std': np.mean(model_df['wrong_std'])}
        row = pd.DataFrame(dict, index=[0])
        df_results = pd.concat([df_results, row], ignore_index=True)
    out_path = os.path.join(save_dir, 'results_mean.csv')
    df_results.to_csv(out_path)


def save_train_val_curve(save_dir, hist):
    metrics = ['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']

    for metric in metrics:
        out_path = os.path.join(save_dir, metric + '.npy')
        arr = np.array(hist.history[metric])
        np.save(out_path, arr)

def get_bag_statistics(instance_ds):
    data = tfds.as_numpy(instance_ds)
    instances_in_bags = [0,0,0,0,0,0,0,0,0,0]
    for bag in data:
        x_bag, y_bag = bag
        sum = np.sum(y_bag)
        instances_in_bags[sum] += 1
    print(instances_in_bags)


