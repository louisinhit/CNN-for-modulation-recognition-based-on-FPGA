#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 00:46:35 2018

@author: lyndon
"""
import os,random
import numpy as np
from numpy import empty
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, random, sys
import h5py
import tensorflow as tf

#from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, ZeroPadding3D, Conv1D
#import tensorlayer as tl
#%%
with open('tensorflow_output.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()

lines = np.hstack(lines)

lines = lines.astype(np.float)
print (lines)

#inpu = [0 for x in range(11)]
#for i in range(11):
#    inpu[i] = float(lines[i])/(2**21)
#    
#inpu = np.vstack(inpu)
#inpu = inpu.reshape((1,11))
#print (inpu)

sm = tf.nn.softmax(lines)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
outputs = sess.run([sm])
print (outputs)
