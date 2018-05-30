
import cv2
import numpy as np

import csv
from matplotlib import cm
from matplotlib import pyplot as plt

from os import path
import cPickle as pickle
import glob
import os

def load_data(load_from_folder, test_split=0.2, num_components=50,
              save_to_file=None, plot_samples=False, seed=113):

    X = []
    labels = []
    samples=[]
    load_from_folder="C:\Users\suyash.a\Desktop\Mosaic\TM\datasets1"
    os.chdir(load_from_folder)
    filist=[]
    for fi in glob.glob("*.pkl"):
        filist.append(fi)
        print(fi)
    for load_from_file in filist:
        if not path.isfile(load_from_file):
            print "Could not find file", load_from_file
            return (X, labels), (X, labels), None, None
        else:
            print "Loading data from", load_from_file
            f = open(load_from_file, 'rb')
            samples = samples+pickle.load(f)
            #print samples[:5]
            labels+=pickle.load(f)
            print labels[:3]
    print "Loaded", len(samples), "training samples"

    X, V, m = extract_features(samples, num_components=num_components)

    if plot_samples:
        print "Plotting samples not implemented"
    print "yo"
    os.chdir("../")
    print os.getcwd()
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)


    X_train = X[:int(len(X)*(1-test_split))]
    y_train = labels[:int(len(X)*(1-test_split))]

    X_test = X[int(len(X)*(1-test_split)):]
    y_test = labels[int(len(X)*(1-test_split)):]

    if save_to_file is not None:

        f = open(save_to_file, 'wb')
        pickle.dump(X_train, f)
        pickle.dump(y_train, f)
        pickle.dump(X_test, f)
        pickle.dump(y_test, f)
        pickle.dump(V, f)
        pickle.dump(m, f)
        f.close()
        print "Save preprocessed data to", save_to_file

    return (X_train, y_train), (X_test, y_test), V, m


def load_from_file(file):

    if path.isfile(file):

        f = open(file, 'rb')
        X_train = pickle.load(f)
        y_train = pickle.load(f)
        X_test = pickle.load(f)
        y_test = pickle.load(f)
        V = pickle.load(f)
        m = pickle.load(f)
        f.close()

    return (X_train, y_train), (X_test, y_test), V, m


def extract_features(X, V=None, m=None, num_components=None):
    if V is None or m is None:

        if num_components is None:
            num_components = 50


        Xarr = np.squeeze(np.array(X).astype(np.float32))


        m, V = cv2.PCACompute(Xarr)


        V = V[:(num_components)]


    for i in xrange(len(X)):
        X[i] = np.dot(V, X[i] - m[0, i])

    return X, V, m
