import numpy as np
import matplotlib.pyplot as plt
import main_functions as main


def get_mini_batch(im_train, label_train, batch_size):
    mini_batch_x, mini_batch_y = None, None
    
    # TO DO
                    
    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    y = None

    # TO DO
    
    return y


def fc_backward(dl_dy, x, w, b, y):
    dl_dx, dl_dw, dl_db = None, None, None

    # TO DO

    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    l, dl_dy = None, None

    # TO DO

    return l, dl_dy

def loss_cross_entropy_softmax(a, y):
    l, dl_da = None, None

    # TO DO

    return l, dl_da

def relu(x):
    y = None

    # TO DO

    return y


def relu_backward(dl_dy, x, y):
    dl_dx = None

    # TO DO

    return dl_dx


def conv(x, w_conv, b_conv):
    y = None

    # TO DO
 
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    dl_dw, dl_db = None, None

    # TO DO
    
    return dl_dw, dl_db

def pool2x2(x):
    y = None

    # TO DO

    return y

def pool2x2_backward(dl_dy, x, y):
   dl_dx = None
   
   # TO DO

   return dl_dx


def flattening(x):
    y = None

    # TO DO

    return y


def flattening_backward(dl_dy, x, y):
    dl_dx = None

    # TO DO

    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    w, b = None, None

    # TO DO
    
    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    w, b = None, None

    # TO DO

    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    w1, b1, w2, b2 = None, None, None, None

    # TO DO

    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    w_conv, b_conv, w_fc, b_fc = None, None, None, None

    # TO DO

    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear(load_weights=False)
    main.main_slp(load_weights=False)
    main.main_mlp(load_weights=False)
    main.main_cnn(load_weights=False)
