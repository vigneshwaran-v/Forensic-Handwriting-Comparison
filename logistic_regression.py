# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 20:21:08 2018

@author: Vicky
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

import pandas as pd
import numpy as np
import os, time, glob

import tensorflow as tf
from sklearn.model_selection import train_test_split

from os import listdir
from os.path import isfile, join
import cv2
import itertools
import random
from sklearn.cluster import KMeans

DATA_SIZE = 100000

print("Imported Requirements")
## Import images
mypath='MLTestData/'
onlyfiles = sorted([ f for f in listdir(mypath) if isfile(join(mypath,f)) ])
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread( join(mypath,onlyfiles[n]),cv2.IMREAD_GRAYSCALE )
print("Imported Images")

## Crop images ##
images = [i[~np.all(i == 255, axis=1)][:,~np.all(i[~np.all(i == 255, axis = 1)] == 255, axis=0)] for i in images]
print("Cropped Images")

## Resize images ##
shapes = [i.shape for i in images]
height = [i[0] for i in shapes]
length = [i[1] for i in shapes]
avg_height = int(sum(height)/len(height))
avg_length = int(sum(length)/len(length))
print(avg_length, avg_height)
images = [cv2.resize(images[i],(avg_length, avg_height)) for i in range(len(images))]
print("Resized Images")

## Write Processed images ##
processed_directory = "Processed/"
subtracted_directory = "Subtracted/"
if not os.path.exists(processed_directory):
    os.makedirs(processed_directory)
if not os.path.exists(subtracted_directory):
    os.makedirs(subtracted_directory)
save = [cv2.imwrite(processed_directory+onlyfiles[i],images[i]) for i in range(len(images))]
print("Processed and Written Images")

## Selecting similar sample ##
authors = sorted(set([i.split("_")[0][:4] for i in onlyfiles]))
author_dict = {}

for i in authors:
    author_dict[i] = []
continue_index = 0
sorted_keys = sorted(author_dict.keys())
for i in sorted_keys:
    for j in onlyfiles[continue_index:]:
        if i in j:
            author_dict[i].append(j)
        else:
            continue_index = onlyfiles.index(j)
            break

permutated_dict = {}
for i in authors:
    permutated_dict[i] = []
for i in author_dict.keys():
    for r in itertools.product(author_dict[i], author_dict[i]):
        if (r[0] != r[1]) and ([r[1],r[0]] not in permutated_dict[i]):
            permutated_dict[i].append([r[0],r[1]])
            
similar = [j for i in permutated_dict.keys() for j in permutated_dict[i]]
similar_sample = random.sample(similar,int(DATA_SIZE/2) if int(DATA_SIZE/2) < len(similar) else len(similar))
print("Similar Sample generated")

## Selecting different sample ##
different_keys = [i.split("*_*") for i in set(["*_*".join(sorted([i,j])) for i,j in itertools.product(author_dict.keys(),author_dict.keys()) if i!=j])]
diff_keys_sample = random.sample(different_keys,int(DATA_SIZE/2) if int(DATA_SIZE/2) < len(different_keys) else len(different_keys))
different_sample = [[random.choice(author_dict[i[0]]),random.choice(author_dict[i[1]])] for i in diff_keys_sample]
print("Different Sample generated")

similar_sample_data = [(images[onlyfiles.index(i[0])],images[onlyfiles.index(i[1])]) for i in similar_sample]
print ("Similar sample done")

different_sample_data = [(images[onlyfiles.index(i[0])],images[onlyfiles.index(i[1])]) for i in different_sample]
print ("different sample data generated")

## Save Not_subtracted data-- The final image pairs ##
not_subtracted_directory = "Not_Subtracted/"
if not os.path.exists(not_subtracted_directory):
    os.makedirs(not_subtracted_directory)
save = [cv2.imwrite(not_subtracted_directory+str(i)+ "$0___" +similar_sample[i][0].split(".")[0]+"__"+similar_sample[i][1],similar_sample_data[i][0]) for i in range(len(similar_sample_data))]
save = [cv2.imwrite(not_subtracted_directory+str(i)+ "$1___" +similar_sample[i][0].split(".")[0]+"__"+similar_sample[i][1],similar_sample_data[i][1]) for i in range(len(similar_sample_data))]
save = [cv2.imwrite(not_subtracted_directory+str(50000+i)+ "$0___" +different_sample[i][0].split(".")[0]+"__"+different_sample[i][1],different_sample_data[i][0]) for i in range(len(different_sample_data))]
save = [cv2.imwrite(not_subtracted_directory+str(50000+i)+ "$1___" +different_sample[i][0].split(".")[0]+"__"+different_sample[i][1],different_sample_data[i][1]) for i in range(len(different_sample_data))]
print("Not Subtracted data saved")

## Import images for SIFT keypoints generation
sift_source_path='Not_Subtracted/'
sift_source_files = sorted([ f for f in listdir(sift_source_path) if isfile(join(sift_source_path,f)) ])
sift_source_images = np.empty(len(sift_source_files), dtype=object)
for n in range(0, len(sift_source_files)):
    sift_source_images[n] = cv2.imread( join(sift_source_path,sift_source_files[n]),cv2.IMREAD_GRAYSCALE )
print("Imported sift source Images")

sift_keypoints_directory = "sift_keypoints/"
if not os.path.exists(sift_keypoints_directory):
    os.makedirs(sift_keypoints_directory)
 
#Generating sift keypoints and descriptors
sift_keypoints=[]
sift_descriptors=[]
for i in range(len(sift_source_files)):
    sift_img = cv2.imread(join(sift_source_path,sift_source_files[i]))
    sift_gray= cv2.cvtColor(sift_img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    sift_kp, sift_des = sift.detectAndCompute(sift_img,None)
    sift_keypoints.append(sift_kp)
    sift_descriptors.append(sift_des)

print("Sift keypoints and descriptors generated")

#Calculating the clusters using KMeans
km = KMeans(n_clusters=30, init='k-means++', n_init=1, max_iter=10, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
concatenated = []
for i in sift_descriptors[:10000]  + sift_descriptors[100000:110000]:
    concatenated += list(i)

cluster_X = np.array(concatenated)
km.fit(cluster_X)
cluster_index = list(y)
desc = sift_descriptors[:10000]  + sift_descriptors[100000:110000]
images = []
for i in desc:
    image = [0 for k in range(30)]
    for j in range(i.shape[0]):
        index = cluster_index.pop()
        image[index] += 1
    images.append(image)

#Creating the data and labels
X_data = []
for i in range(0,20000,2):
    #print(i,i+1)
    X_data.append(abs(np.array(images[i])-np.array(images[i+1])))

X_data = np.array(X_data)
Y_data = np.array([1 for i in range(5000)] + [0 for i in range(5000)])

### Splitting the data into training and testing ###
train_data = {}
test_data = {}
train_data["x"], test_data["x"], train_data["y"], test_data["y"] = train_test_split(X_data, Y_data, test_size=0.2)

MAX_ITER = 10000
LEARNING_RATE = 0.0005

#Logistic regression function defenition
def train_logistic_regression(train_X,train_Y,max_iter,learn_rate):
    from sklearn.linear_model import LogisticRegression
    logisticRegr = LogisticRegression(solver = 'sag', multi_class="ovr", max_iter=max_iter, C=learn_rate)
    logisticRegr.fit(train_X, train_Y)
    return logisticRegr
#Function to calculate accuracy of the model
def accuracy_logistic_regression(trained_model, test_X, test_Y):
    predictions = trained_model.predict(test_X)
    score = trained_model.score(test_X, test_Y)
    return score

#training the model and printing the accuracy
trained_model = train_logistic_regression(train_data["x"],train_data["y"],MAX_ITER,LEARNING_RATE)
print("Accuracy on AND Testing Data: " + str(accuracy_logistic_regression(trained_model,test_data["x"],test_data["y"])))

test_data_type1 = {"x":[],"y":[]}
test_data_type2 = {"x":[],"y":[]}


#calculating Type 1 & Type 2Error
for i in range(2000):
    if(test_data["y"][i]==0):
        test_data_type1["x"].append(test_data["x"][i])
        test_data_type1["y"].append(test_data["y"][i])
    else:
        test_data_type2["x"].append(test_data["x"][i])
        test_data_type2["y"].append(test_data["y"][i])

print("Type1 Error: " + str(1-accuracy_logistic_regression(trained_model,test_data_type1["x"],test_data_type1["y"])))
print("Type2 Error: " + str(1-accuracy_logistic_regression(trained_model,test_data_type2["x"],test_data_type2["y"])))
