import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from p4 import get_mini_batch, fc, relu, conv, pool2x2, flattening
from p4 import train_slp_linear, train_slp, train_mlp, train_cnn


def main_slp_linear(dataset_dir='./ReducedMNIST/', load_weights=False):
    train_data_path = os.path.join(dataset_dir, 'mnist_train.mat')
    test_data_path = os.path.join(dataset_dir, 'mnist_test.mat')
    mnist_train = sio.loadmat(train_data_path)
    mnist_test = sio.loadmat(test_data_path)
    im_train, label_train = mnist_train['im_train'], mnist_train['label_train']
    im_test, label_test = mnist_test['im_test'], mnist_test['label_test']
    batch_size = 32
    im_train, im_test = im_train / 255.0, im_test / 255.0
    mini_batch_x, mini_batch_y = get_mini_batch(im_train, label_train, batch_size)

    if load_weights:
        data = np.load('slp_linear.npz')
        w, b = data['w'], data['b']
    else:
        w, b = train_slp_linear(mini_batch_x, mini_batch_y)
        np.savez('slp_linear.npz', w=w, b=b)

    acc = 0
    confusion = np.zeros((10, 10))
    num_test = im_test.shape[1]
    for i in range(num_test):
        x = im_test[:, [i]]
        y = fc(x, w, b)
        l_pred = np.argmax(y)
        confusion[l_pred, label_test[0, i]] = confusion[l_pred, label_test[0, i]] + 1

        if l_pred == label_test[0, i]:
            acc = acc + 1
    accuracy = acc / num_test
    for i in range(10):
        confusion[:, i] = confusion[:, i] / np.sum(confusion[:, i])

    label_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    visualize_confusion_matrix(confusion, accuracy, label_classes, 'Single-layer Linear Perceptron Confusion Matrix')

def main_slp(dataset_dir='./ReducedMNIST/', load_weights=False):
    train_data_path = os.path.join(dataset_dir, 'mnist_train.mat')
    test_data_path = os.path.join(dataset_dir, 'mnist_test.mat')
    mnist_train = sio.loadmat(train_data_path)
    mnist_test = sio.loadmat(test_data_path)
    im_train, label_train = mnist_train['im_train'], mnist_train['label_train']
    im_test, label_test = mnist_test['im_test'], mnist_test['label_test']
    batch_size = 32
    im_train, im_test = im_train / 255.0, im_test / 255.0
    mini_batch_x, mini_batch_y = get_mini_batch(im_train, label_train, batch_size)

    if load_weights:
        data = np.load('slp.npz')
        w, b = data['w'], data['b']
    else:
        w, b = train_slp(mini_batch_x, mini_batch_y)
        np.savez('slp.npz', w=w, b=b)

    acc = 0
    confusion = np.zeros((10, 10))
    num_test = im_test.shape[1]
    for i in range(num_test):
        x = im_test[:, [i]]
        y = fc(x, w, b)
        l_pred = np.argmax(y)
        confusion[l_pred, label_test[0, i]] = confusion[l_pred, label_test[0, i]] + 1

        if l_pred == label_test[0, i]:
            acc = acc + 1
    accuracy = acc / num_test
    for i in range(10):
        confusion[:, i] = confusion[:, i] / np.sum(confusion[:, i])

    label_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    visualize_confusion_matrix(confusion, accuracy, label_classes, 'Single-layer Perceptron Confusion Matrix')

def main_mlp(dataset_dir='./ReducedMNIST/', load_weights=False):
    train_data_path = os.path.join(dataset_dir, 'mnist_train.mat')
    test_data_path = os.path.join(dataset_dir, 'mnist_test.mat')
    mnist_train = sio.loadmat(train_data_path)
    mnist_test = sio.loadmat(test_data_path)
    im_train, label_train = mnist_train['im_train'], mnist_train['label_train']
    im_test, label_test = mnist_test['im_test'], mnist_test['label_test']
    batch_size = 32
    im_train, im_test = im_train / 255.0, im_test / 255.0
    mini_batch_x, mini_batch_y = get_mini_batch(im_train, label_train, batch_size)

    if load_weights:
        data = np.load('mlp.npz')
        w1, b1, w2, b2 = data['w1'], data['b1'], data['w2'], data['b2']
    else:
        w1, b1, w2, b2 = train_mlp(mini_batch_x, mini_batch_y)
        np.savez('mlp.npz', w1=w1, b1=b1, w2=w2, b2=b2)

    acc = 0
    confusion = np.zeros((10, 10))
    num_test = im_test.shape[1]
    for i in range(num_test):
        x = im_test[:, [i]]
        pred1 = fc(x, w1, b1)
        pred2 = relu(pred1)
        y = fc(pred2, w2, b2)
        l_pred = np.argmax(y)
        confusion[l_pred, label_test[0, i]] = confusion[l_pred, label_test[0, i]] + 1

        if l_pred == label_test[0, i]:
            acc = acc + 1
    accuracy = acc / num_test
    for i in range(10):
        confusion[:, i] = confusion[:, i] / np.sum(confusion[:, i])

    label_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    visualize_confusion_matrix(confusion, accuracy, label_classes, 'Multi-layer Perceptron Confusion Matrix')


def main_cnn(dataset_dir='./ReducedMNIST/', load_weights=False):
    train_data_path = os.path.join(dataset_dir, 'mnist_train.mat')
    test_data_path = os.path.join(dataset_dir, 'mnist_test.mat')
    mnist_train = sio.loadmat(train_data_path)
    mnist_test = sio.loadmat(test_data_path)
    im_train, label_train = mnist_train['im_train'], mnist_train['label_train']
    im_test, label_test = mnist_test['im_test'], mnist_test['label_test']
    batch_size = 32
    im_train, im_test = im_train / 255.0, im_test / 255.0
    mini_batch_x, mini_batch_y = get_mini_batch(im_train, label_train, batch_size)

    if load_weights:
        data = np.load('cnn.npz')
        w_conv, b_conv, w_fc, b_fc = data['w_conv'], data['b_conv'], data['w_fc'], data['b_fc']
    else:
        w_conv, b_conv, w_fc, b_fc = train_cnn(mini_batch_x, mini_batch_y)
        np.savez('cnn.npz', w_conv=w_conv, b_conv=b_conv, w_fc=w_fc, b_fc=b_fc)
    
    acc = 0
    confusion = np.zeros((10, 10))
    num_test = im_test.shape[1]
    for i in range(num_test):
        x = im_test[:, [i]].reshape((14, 14, 1), order='F')
        pred1 = conv(x, w_conv, b_conv)  # (14, 14, 3)
        pred2 = relu(pred1)  # (14, 14, 3)
        pred3 = pool2x2(pred2)  # (7, 7, 3)
        pred4 = flattening(pred3)  # (147, 1)
        y = fc(pred4, w_fc, b_fc)  # (10, 1)
        l_pred = np.argmax(y)
        confusion[l_pred, label_test[0, i]] = confusion[l_pred, label_test[0, i]] + 1
        if l_pred == label_test[0, i]:
            acc = acc + 1
    accuracy = acc / num_test
    for i in range(10):
        confusion[:, i] = confusion[:, i] / np.sum(confusion[:, i])

    label_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    visualize_confusion_matrix(confusion, accuracy, label_classes, 'CNN Confusion Matrix')


def visualize_confusion_matrix(confusion, accuracy, label_classes, name):
    plt.title("{}, accuracy = {:.3f}".format(name, accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.show()
