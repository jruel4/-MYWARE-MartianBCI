# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 18:52:25 2017

@author: marzipan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from Demos import RL_Environment_Simulator as RLEnv
from random import randint
import random
from sklearn import datasets
from matplotlib import pyplot as plt

# import some data to play with
iris = datasets.load_iris()

# Data sets
IRIS_TRAINING = "cs-testing.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)


env = RLEnv.RLEnv()
x = np.asarray([randint(0,50) for i in range(100000)])
x = x.reshape((x.shape[0],1))
y = np.asarray([int(env.interact([n,n])[0][0]) for n in x])



# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10,20,10],
                                            n_classes=21)

# Fit model.
classifier.fit(x=x,
               y=y,
               steps=20000)



# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=x[3000:3500],
                                     y=y[3000:3500])["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.
new_samples = np.array(
    [[randint(1,3)] for i in range(50)], dtype=float)

y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))
