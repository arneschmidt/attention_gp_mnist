import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(bag_model, instance_model, test_gen_instances, save_dir):
    n_visualize = 100
    test_data = test_gen_instances.as_numpy_iterator()

    n = len(test_gen_instances)
    correct_attention_preds = 0
    pos_bags = 0
    for i in range(n):
        x_bag, y_bag = next(test_data)
        bag_pred = bag_model.predict(x_bag)
        preds = instance_model.predict(x_bag)
        correct_attention_preds += _correct_attention(preds, y_bag)
        pos_bags += np.max(y_bag)
        if i < n_visualize:
            _plot_images(x_bag, bag_pred, preds, os.path.join(save_dir + str(i)) + '.jpg')
    attention_accuracy = correct_attention_preds/pos_bags
    print('Attention accuracy: ',  str(attention_accuracy))

def _plot_images(x_bag, bag_pred, preds, save_path):
    f, axarr = plt.subplots(3, 3)
    f.suptitle('Bag pred probability: ' + str(bag_pred[0][1]), fontsize=16)
    for i in range(3):
        for j in range(3):
            n = (i*3) + j
            axarr[i, j].imshow(x_bag[n], cmap='gray_r')
            axarr[i, j].set_title('Attention: ' + str(np.round(preds[n], 2)))
    f.tight_layout(pad=1.5)
    plt.savefig(save_path)
    plt.close()

def _correct_attention(instance_preds, instance_gt):
    am = np.argmax(instance_preds)
    if instance_gt[am] == 1:
        return 1
    else:
        return 0

