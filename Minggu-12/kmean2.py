import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import time
import itertools

datasetPath = "D:\Project\Mall_Customers1.txt"
dataset = np.loadtxt(datasetPath, delimiter=" ")

k = 2
iterationCounter = 0
input = dataset

def initCentroid(dataIn, k):
    result = dataIn[np.random.choice(dataIn.shape[0], k, replace=False)]
    return result