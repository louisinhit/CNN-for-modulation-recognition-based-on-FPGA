#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 22:43:41 2018

@author: xueyuan

this file expand the weight and bias by 2^21
note that in ubuntu newline is '\n'
but in Windows it should be '\r\n'
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
with open('conv1w.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()

lines = np.vstack(lines)
print (lines.shape)
print (np.max(lines.astype(np.float)))
print (np.min(lines.astype(np.float)))
f= open("conv1w_.txt","w+")
for a in lines:
    f.write("%d\n" % int(float(a)*(2**21)))
f.close()

f= open("conv1w_cc.txt","w+")
for a in lines:
    f.write("%f\n" % float(int(float(a)*(2**21))/(2**21)))
f.close()

with open('conv1b.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()

lines = np.vstack(lines)
print (lines.shape)
print (np.max(lines.astype(np.float)))
print (np.min(lines.astype(np.float)))
f= open("conv1b_.txt","w+")
for a in lines:
    f.write("%d\n" % int(float(a)*(2**21)))
f.close()

f= open("conv1b_cc.txt","w+")
for a in lines:
    f.write("%f\n" % float(int(float(a)*(2**21))/(2**21)))
f.close()
#%%
with open('conv2w.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()

lines = np.vstack(lines)
print (lines.shape)
print (np.max(lines.astype(np.float)))
print (np.min(lines.astype(np.float)))
f= open("conv2w_.txt","w+")
for a in lines:
    f.write("%d\n" % int(float(a)*(2**21)))
f.close()

f= open("conv2w_cc.txt","w+")
for a in lines:
    f.write("%f\n" % float(int(float(a)*(2**21))/(2**21)))
f.close()

with open('conv2b.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()

lines = np.vstack(lines)
print (lines.shape)
print (np.max(lines.astype(np.float)))
print (np.min(lines.astype(np.float)))
f= open("conv2b_.txt","w+")
for a in lines:
    f.write("%d\n" % int(float(a)*(2**21)))
f.close()
f= open("conv2b_cc.txt","w+")
for a in lines:
    f.write("%f\n" % float(int(float(a)*(2**21))/(2**21)))
f.close()
#%%
with open('den1w.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()

lines = np.vstack(lines)
print (lines.shape)
print (np.max(lines.astype(np.float)))
print (np.min(lines.astype(np.float)))
f= open("den1w_.txt","w+")
for a in lines:
    f.write("%d\n" % int(float(a)*(2**21)))
f.close()
f= open("den1w_cc.txt","w+")
for a in lines:
    f.write("%f\n" % float(int(float(a)*(2**21))/(2**21)))
f.close()

with open('den1b.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()

lines = np.vstack(lines)
print (lines.shape)
print (np.max(lines.astype(np.float)))
print (np.min(lines.astype(np.float)))
f= open("den1b_.txt","w+")
for a in lines:
    f.write("%d\n" % int(float(a)*(2**21)))
f.close()
f= open("den1b_cc.txt","w+")
for a in lines:
    f.write("%f\n" % float(int(float(a)*(2**21))/(2**21)))
f.close()
#%%
with open('den2w.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()

lines = np.vstack(lines)
print (lines.shape)
print (np.max(lines.astype(np.float)))
print (np.min(lines.astype(np.float)))
f= open("den2w_.txt","w+")
for a in lines:
    f.write("%d\n" % int(float(a)*(2**21)))
f.close()
f= open("den2w_cc.txt","w+")
for a in lines:
    f.write("%f\n" % float(int(float(a)*(2**21))/(2**21)))
f.close()
with open('den2b.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()

lines = np.vstack(lines)
print (lines.shape)
print (np.max(lines.astype(np.float)))
print (np.min(lines.astype(np.float)))
f= open("den2b_.txt","w+")
for a in lines:
    f.write("%d\n" % int(float(a)*(2**21)))
f.close()
f= open("den2b_cc.txt","w+")
for a in lines:
    f.write("%f\n" % float(int(float(a)*(2**21))/(2**21)))
f.close()