# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 24:00:00 2017

@author: Zhiyu
"""
# import your module here
import numpy as np
import time
from threading import Thread
import functools
import math

# (global) variable definition here
TRAINING_TIME_LIMIT = 60*10
global A,distanceMatrix,rawData,label,label2indexs
# class definition here

# function definition here
def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

@timeout(TRAINING_TIME_LIMIT)
def train(traindata):
    init(traindata)
    print(updateDistanceMatrix())
    # 在此处完成你的训练函数，注意训练时间不要超过TRAINING_TIME_LIMIT(秒)。
    time.sleep(1) # 这行仅用于测试训练超时，运行时请删除这行，否则你的TRAINING_TIME_LIMIT将-1s。
    return 0

def init(traindata):
    global rawData,A,label,distanceMatrix
    rawData=traindata[0]
    label=traindata[1]
    A=np.eye(rawData.shape[1])
    distanceMatrix = np.zeros((len(label), len(label)))
    classifyLabel(label)

def Euclidean_distance(inst_a, inst_b):
    return np.linalg.norm(inst_a - inst_b)

def distance(inst_a, inst_b):           #行向量
    dist = np.dot(A,(inst_a-inst_b).transpose())
    return np.dot(dist.transpose(),dist)    #(A(x-y))T(A(x-y))

def classifyLabel(label):
    global label2indexs
    label2indexs={}
    for i in range(len(label)):
        if label[i] in label2indexs.keys():
            label2indexs[label[i]].append(i)
        else:
            label2indexs[label[i]]=[i]
    return label2indexs

def updateDistanceMatrix():
    global distanceMatrix,rawData
    for i in range(len(rawData)):
        for j in range(i+1):
            distanceMatrix[i,j]=math.exp(-distance(rawData[i],rawData[j]))
            distanceMatrix[j,i]=distanceMatrix[i,j]
    return distanceMatrix

def Pij(i,j):
    global distanceMatrix
    return distanceMatrix[i,j]/(sum(distanceMatrix[i,:])-1)

def Pi(i):
    global distanceMatrix,label,label2indexs
    indexs=label2indexs[label[i]]
    pi=0
    for j in indexs:
        pi+=Pij(i,j)
    return pi

def deltaA():


# main program here
if  __name__ == '__main__':
    pass