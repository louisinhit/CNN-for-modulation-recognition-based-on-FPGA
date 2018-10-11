
import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, ZeroPadding3D, Conv1D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, random, sys, keras
import h5py

from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.models import load_model

import tensorflow as tf
#%%
mod = tf.keras.models.load_model('convmod_CNN2.wts.h5')

conv1w = mod.get_weights()[0]
conv1b = mod.get_weights()[1]
conv2w = mod.get_weights()[2]
conv2b = mod.get_weights()[3]
den1w = mod.get_weights()[4]
den1b = mod.get_weights()[5]
den2w = mod.get_weights()[6]
den2b = mod.get_weights()[7]

f= open("conv1w.txt","w+")
for n in range(64):
    for h in range(3):
        for w in range(3):
            a = conv1w[h,w,:,n]
            f.write("%f\n" % (a))
f.close()
print (conv1w)
f= open("conv1b.txt","w+")
for inp in range(64):
    a = conv1b[inp]
    f.write("%f\n" % (a))
f.close()
f= open("conv2w.txt","w+")
for n in range(40):
    for h in range(3):
        for w in range(3):
            for inp in range(64):
                a = conv2w[h,w,inp,n]
                f.write("%f\n" % (a))
f.close()
f= open("conv2b.txt","w+")
for n in range(40):     
    a = conv2b[n]
    f.write("%f\n" % (a))
f.close()
f= open("den1w.txt","w+")
for out in range(128):
    for inp in range(31360):
        a = den1w[inp,out]
        f.write("%f\n" % (a))
f.close()
f= open("den1b.txt","w+")
for inp in range(128):
    a = den1b[inp]
    f.write("%f\n" % (a))
f.close()
f= open("den2w.txt","w+")
for n in range(11):
    for inp in range(128):
        a = den2w[inp,n]
        f.write("%f\n" % (a))
f.close()
f= open("den2b.txt","w+")
for n in range(11):
    a = den2b[n]
    f.write("%f\n" % (a))
f.close()