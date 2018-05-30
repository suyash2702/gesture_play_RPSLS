

import cv2
import numpy as np

from datasets1 import homebrew
from classifiers1 import MultiLayerPerceptron




def main():
    (X_train, y_train), (X_test, y_test), _, _ = homebrew.load_data(
        load_from_folder="/datasets1",
        num_components=50,
        test_split=0.2,
        save_to_file="datasets1/faces_preprocessed.pkl",
        seed=42)
    if len(X_train) == 0 or len(X_test) == 0:
        print "Empty data"
        raise SystemExit

    X_train = np.squeeze(np.array(X_train)).astype(np.float32)
    y_train = np.array(y_train)
    X_test = np.squeeze(np.array(X_test)).astype(np.float32)
    y_test = np.array(y_test)


    labels = np.unique(np.hstack((y_train, y_test)))
    print X_train.shape

    num_features = len(X_train[0])
    print num_features
    num_classes = len(labels)
    print num_classes
    params = dict(term_crit=(cv2.TERM_CRITERIA_COUNT, 300, 0.01),
                  train_method=cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
                  bp_dw_scale=0.001, bp_moment_scale=0.9)
    saveFile = 'params1/mlp.xml'

    print "---"
    print "1-hidden layer networks"
    best_acc = 0.0  
    for l1 in xrange(20):
        layerSizes = np.int32([num_features, (l1 + 1) * num_features/5,(l1+1)*num_features/10,num_classes])
        MLP = MultiLayerPerceptron(layerSizes, labels)
        print layerSizes
        MLP.fit(X_train, y_train, params=params)
        (acc, _, _) = MLP.evaluate(X_train, y_train)
        print " - train acc = ", acc
        (acc, _, _) = MLP.evaluate(X_test, y_test)
        print " - test acc = ", acc
        if acc > best_acc:
            MLP.save(saveFile)
            best_acc = acc


if __name__ == '__main__':
    main()
