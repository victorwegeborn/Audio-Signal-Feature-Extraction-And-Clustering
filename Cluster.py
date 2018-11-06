import argparse
import importlib
import itertools
import numpy as np
from time import time
from DataHandler import DataHandler

import imp
from scipy.stats import mode
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        help='Configuration filename located in ./configuration/',
        required=True
    )
    args = parser.parse_args()


    # Import configuration file and set subdirectories for data loading
    # if --load_all is set load_directories are overwritten by subdirs in data_path
    config = importlib.import_module('configuration.%s' % args.config)

    # convert subdirectories and mergings into a filename
    # to store data after preproccessing (feature extraction)
    print(82 * '_')
    print('Loading training data...\n')
    data, targets = DataHandler().load_data(config.classes,
                                            config.features)

    print('\t\t{}\t{}'.format('n samples', 'features/classes'))
    print('data:\t\t{}\t\t{}\ntargets:\t{}\t\t{}'.format(
                data.shape[0], data.shape[1], targets.shape[0], set(np.unique(targets))))

    # Transforms features by scaling each feature between (0,1).
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    print(82 * '_')
    print('Clustering data...\n')
    print('Algo\t\ttime\tinertia\thomo\tcompl\tv-meas')
    clusters = evaluate(KMeans(init='k-means++', n_clusters=len(set(np.unique(targets))),
                               n_init=200, max_iter=10000),
                               name='k-means', data=data,
                               labels=targets.squeeze())

    # Remap targets/labels after clustering
    # in order to be able to compute accuracy
    # and display confusion matrix
    cluster_targets = np.zeros_like(targets)
    for i in np.unique(targets):
        mask = (clusters == i)
        cluster_targets[mask] = mode(targets[mask])[0]

    print('\nModel accuracy: %.3f' % (metrics.accuracy_score(targets, cluster_targets)))
    cm = metrics.confusion_matrix(targets, cluster_targets)
    plot_confusion_matrix(cm=cm,
                          classes=config.classes,
                          normalize=True,
                          title='Confusion matrix (normalized)')


def evaluate(estimator, name, data, labels):
    t0 = time()
    clusters = estimator.fit_predict(data)
    print('%-9s\t%.2fs\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_)))
    return clusters



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    This function was taken from the confusion matrix example at sklean website:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    '''

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
