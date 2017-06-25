# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp

from LoadXDF import get_raw_data

[ts,locs] = get_raw_data()

raw = tf.constant(ts)
