import os
import matplotlib.pyplot as plt
import numpy as np


def visualize_attention(bag_level_uncertainty_model, instance_model, test_gen_instances, save_dir, quick_eval=True):
    n_visualize = 100
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

    print('Bag Accuracy: ' + str(accuracy)
          + '; correct_pred_std: ' + str(correct_pred_std)
          + '; wrong_pred_std: ' + str(wrong_pred_std))

