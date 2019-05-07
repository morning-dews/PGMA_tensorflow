import numpy as np
import copy 
import pickle
import os
from tensorflow.examples.tutorials.mnist import input_data

class mnist_data(object):
    def __init__(self, x, y, x_, y_):
        self.data = x.reshape((-1, 28, 28, 1))
        self.labels = y
        self.test_data = x_.reshape((-1, 28, 28, 1))
        self.test_labels = y_
        self.data_shape = (28, 28, 1)
        self.num_points = len(x)
        self.test_num_points =len(x_)

def set_data(dataset, task_num):

    x, y, x_, y_ = mnist(dataset, task_num)

    noOfTask = len(x)
    data = []
    for i in range(noOfTask):
        data.append(mnist_data(x[i], y[i], x_[i], y_[i]))

       
    return data

def mnist(dataset, task_num):
    """
    Load Dataset and set package.
    """
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    if dataset == 'shuffle_mnist':
        x = []
        x_ = []
        y = []
        y_ = []

        x.append(np.concatenate((mnist.train.images, mnist.validation.images)))
        y.append(np.concatenate((mnist.train.labels, mnist.validation.labels)))
        x_.append(mnist.test.images)
        y_.append(mnist.test.labels)
        
        for i in range(1, task_num):
            
            np.random.seed(451)
            idx = np.arange(784)                 # indices of shuffling image
            np.random.shuffle(idx)
            
            # x.append(x[i-1].copy())
            # x_.append(x_[i-1].copy())
            # y.append(y[i-1].copy())
            # y_.append(y_[i-1].copy())
            x.append(copy.deepcopy(x[i-1]))
            x_.append(copy.deepcopy(x_[i-1]))
            y.append(copy.deepcopy(y[i-1]))
            y_.append(copy.deepcopy(y_[i-1]))

            x[i] = x[i][:,idx]           # applying to shuffle
            x_[i] = x_[i][:,idx]

        return x, y, x_, y_
    
    if dataset == 'disjoint_mnist':

        x = np.concatenate((mnist.train.images,mnist.validation.images))
        y = np.concatenate((mnist.train.labels,mnist.validation.labels))
        x_ = mnist.test.images
        y_ = mnist.test.labels
        task_split = int(10/task_num) 
        task_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        tasks = []
        for i in range(task_num):
            if len(task_labels) >= 2 * task_split:
                tasks.append(task_labels[:task_split])
                task_labels = task_labels[task_split:]
            else:
                tasks.append(task_labels[:])
                task_labels = []
        # print(tasks)
        #############################################
        #training data processing
        train_idx = [[] for i in range(task_num)]
        for i in range(len(y)):
            label_i = np.argmax(y[i,:])
            for j in range(task_num):
                if label_i in tasks[j]:
                    train_idx[j].append(i)
                    break

        train = [[] for i in range(task_num)]
        train_label = [[] for i in range(task_num)]

        for i in range(task_num):
            
            train_idx_t = np.array(train_idx[i])
            train[i] = x[train_idx_t, :]
            train_label[i] = y[train_idx_t, :]

        #############################################
        #test data processing
        test_idx = [[] for i in range(task_num)]
        for i in range(len(y_)):
            label_i = np.argmax(y_[i,:])
            for j in range(task_num):
                if label_i in tasks[j]:
                    test_idx[j].append(i)
                    break

        test = [[] for i in range(task_num)]
        test_label = [[] for i in range(task_num)]

        for i in range(task_num):
            
            test_idx_t = np.array(test_idx[i])
            test[i] = x_[test_idx_t, :]
            test_label[i] = y_[test_idx_t, :]
        
        # print(test)
        # print(test_label)
        return train, train_label, test, test_label