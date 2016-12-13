#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time
import datetime

import numpy as np
import theano
import theano.tensor as T

import lasagne

from scipy import signal

from lasagne.regularization import regularize_layer_params, l2, l1


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
    dir = "mnist_dataset/"
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, dir + filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images(dir + 'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(dir + 'train-labels-idx1-ubyte.gz')
    X_test =  load_mnist_images(dir + 't10k-images-idx3-ubyte.gz')
    y_test =  load_mnist_labels(dir + 't10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def prune(layer, threshold, prnt=False):
    # get numpy matrix
    values = layer.W.get_value()

    mask = np.abs(values) > threshold

    if prnt:
        size = values.shape[0] * values.shape[1]
        zeros = np.sum(mask == False) + 0.0
        print("{0:0.2f}% zeros".format(zeros / size * 100.0))

    return mask


def prune_matrix(matrix, percentage):
    values = np.abs(matrix)
    flat_sorted = sorted(values.flatten())

    # find from where to prune
    index = int(round(len(flat_sorted) * percentage))
    threshold = flat_sorted[index]

    return values > threshold


def prune_by_value(layer, percentage, prnt=False):
    return prune_matrix(layer.W.get_value(), percentage)


def prune_by_hessian(layer, loss, percentage):
    #h = theano.gradient.hessian(loss, layer.W)
    h = theano.tensor.grad(loss, layer.W)
    f = theano.function([layer.W], h)
    hessian_val = f(layer.W.get_value())

    return prune_matrix(hessian_val, percentage)


def mask_layer(layer, mask):
    layer.W.set_value(layer.W.get_value() * mask)


def print_zero_num(layer, name=""):
    values = layer.W.get_value()
    size = values.shape[0] * values.shape[1]
    zeros = np.sum(values == 0) 
    perc = 100.0 * zeros / size
    print("  Layer {0}:\t\t{1:0.2f}% pruned".format(name, perc))
    return perc


import theano.tensor as T
def weird(x):
    return T.sum(T.sqrt(T.maximum(T.sum(abs(x), axis=0), 0.001))) + T.sum(T.sqrt(T.maximum(T.sum(abs(x), axis=1), 0.001)))


def build_net(input_var, neurons1, neurons2):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=neurons1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.2)
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=neurons2,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.2)
    network = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network, l_hid1, l_hid2


def main(num_epochs=500, decay=0.0001, percentage=0.95, prune_every=20, learning_rate=0.01, lr_decay=0.99, neighborhood=31):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network, l_hid1, l_hid2 = build_net(input_var, 250, 150)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    penalty_func = l2
    penalty_1 = regularize_layer_params(l_hid1, penalty_func) * decay 
    penalty_2 = regularize_layer_params(l_hid2, penalty_func) * decay 

    loss = loss + penalty_1 + penalty_2

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.

    #lr = theano.shared(np.array(learning_rate, dtype=theano.shared.floatX))
    lr = theano.shared(np.array(learning_rate, dtype=theano.config.floatX))
    #lr_decay = np.array(lr_decay, dtype=theano.config.floatX)
    #lr = theano.shared(learning_rate)

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=lr, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # make masks for weigths
    mask_1 = None; mask_2 = None; mask_3 = None;

    # meta 
    min_error = 85 
    max_error = 0
    underperfomed = 0
    max_underperfomed = 20

    try: 
        # We iterate over epochs:
        for epoch in range(num_epochs):

            lr.set_value((lr.get_value() * lr_decay).astype('float32'))

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

                if mask_1 is not None and mask_2 is not None and mask_3 is not None:
                    mask_layer(l_hid1, mask_1)
                    mask_layer(l_hid2, mask_2)
                    mask_layer(network, mask_3)

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("\nEpoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  learning rate:\t\t{:.6f}".format(float(lr.get_value())))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
            max_error = max(max_error, val_acc / val_batches * 100)
            print("  max validation acc:\t\t{:.2f} %".format(
                max_error))


            # early stopping
            if epoch > 100 and val_acc / val_batches * 100 < min_error:
                underperfomed += 1
                if underperfomed > max_underperfomed:
                    break
            else:
                underperfomed = 0

            # every n epochs, update the mask
            if epoch % prune_every == prune_every - 1:
                mask_1 = prune_by_value(l_hid1, 1 - percentage**(epoch / prune_every + 1), prnt=True)
                mask_2 = prune_by_value(l_hid2, 1 - percentage**(epoch / prune_every + 1), prnt=True)
                mask_3 = prune_by_value(network, 1 - percentage**(epoch / prune_every + 1), prnt=True)
                #mask_1 = prune_by_hessian(l_hid1, loss, 1 - percentage**(epoch / prune_every + 1))
                #mask_2 = prune_by_hessian(l_hid2, loss, 1 - percentage**(epoch / prune_every + 1))

            z1 = print_zero_num(l_hid1, '1')
            z2 = print_zero_num(l_hid2, '2')

    except KeyboardInterrupt:
        pass

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    main(num_epochs=7000, prune_every=10, percentage=0.97, decay=0.0, learning_rate=0.02, lr_decay=0.997, neighborhood=15)
