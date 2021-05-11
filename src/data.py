import matplotlib.pyplot as plt
import numpy as np

def generate_data(number_of_bags = 1000):
    instances_per_bag = 8
    positive_instances_per_bag = 2
    percentage_positive_bags = 0.5
    mean_positive = [[1, 0], [0, 1]]
    mean_negative = [[1, 1], [0, 0]]
    covariance = [[0.01, 0.0], [0.0, 0.01]]
    negative_instances_per_bag = instances_per_bag - positive_instances_per_bag
    np.random.seed(42)
    x = np.zeros([number_of_bags, instances_per_bag, 2])
    y = np.zeros([number_of_bags, instances_per_bag])
    y_bags = np.zeros([number_of_bags, 1])
    y_bags[0:int(number_of_bags * percentage_positive_bags), :] = 1
    for i in range(number_of_bags):
        x_neg1 = np.random.multivariate_normal(mean_negative[0], covariance, int(negative_instances_per_bag / 2))
        x_neg2 = np.random.multivariate_normal(mean_negative[1], covariance, int(negative_instances_per_bag / 2))
        y_instances = np.full(shape=(negative_instances_per_bag), fill_value=0)
        x_instances = np.concatenate((x_neg1, x_neg2))

        if i < number_of_bags * percentage_positive_bags:
            x_pos1 = np.random.multivariate_normal(mean_positive[0], covariance, int(positive_instances_per_bag/2))
            x_pos2 = np.random.multivariate_normal(mean_positive[1], covariance, int(positive_instances_per_bag/2))
            y_pos = np.full(shape=(positive_instances_per_bag), fill_value=1)
            x_instances = np.concatenate((x_pos1, x_pos2, x_instances))
            y_instances = np.concatenate((y_pos, y_instances))
        else:
            x_neg1 = np.random.multivariate_normal(mean_negative[0], covariance, int(positive_instances_per_bag / 2))
            x_neg2 = np.random.multivariate_normal(mean_negative[1], covariance, int(positive_instances_per_bag / 2))
            y_neg = np.full(shape=(positive_instances_per_bag), fill_value=0)
            x_instances = np.concatenate((x_neg1, x_neg2, x_instances))
            y_instances = np.concatenate((y_neg, y_instances))
        x[i] = x_instances
        y[i] = y_instances
        # plt.figure()
        # plt.plot(x_instances[...,0], x_instances[...,1], 'kx')
        # plt.show()
    x = np.expand_dims(np.array(x, dtype=np.float), axis=1)
    y = np.array(y, dtype=np.float)
    y_bags = np.array(y_bags, dtype=np.float)

    return x, y, y_bags