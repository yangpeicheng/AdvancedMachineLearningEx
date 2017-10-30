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
global A,distanceMatrix,rawData,label,label2indexs,normValue
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
    global A
    init(traindata)
    lastfa=-100
    for i in range(50):
        print(i)
        updateDistanceMatrix()
        A=A+0.03*deltaA()
        print(A)
        currentfa=fA()
        print(currentfa-lastfa)
        lastfa=currentfa
    # 在此处完成你的训练函数，注意训练时间不要超过TRAINING_TIME_LIMIT(秒)。
    print(A)
    return 0

def init(traindata):
    global rawData,A,label,distanceMatrix,normValue
    rawData=traindata[0]
    label=traindata[1]
    A=np.random.rand(rawData.shape[1],rawData.shape[1])
    distanceMatrix = np.zeros((len(label), len(label)))
    normValue=[0 for i in range(len(label))]
    classifyLabel(label)

def Euclidean_distance(inst_a, inst_b):
    return np.linalg.norm(inst_a - inst_b)

def distance(inst_a, inst_b):           #行向量
    transformed_a=np.dot(A,inst_a.transpose())
    transformed_b=np.dot(A,inst_b.transpose())
    return Euclidean_distance(transformed_a,transformed_b)    #(A(x-y))T(A(x-y))

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
    global distanceMatrix,rawData,normValue
    l=len(rawData)
    for i in range(l):
        for j in range(i+1):
            distanceMatrix[i,j]=math.exp(-math.pow(distance(rawData[i],rawData[j]),2))
            distanceMatrix[j,i]=distanceMatrix[i,j]
    for i in range(l):
        normValue[i]=sum(distanceMatrix[i,:])-1
    return distanceMatrix

def Pij(i,j):
    global distanceMatrix,normValue
    if i==j:
        return 0
    return distanceMatrix[i,j]/normValue[i]

def Pi(i):
    global distanceMatrix,label,label2indexs
    indexs=label2indexs[label[i]]
    pi=0
    for j in indexs:
        pi+=Pij(i,j)
    return pi

def deltaA():
    global label,label2indexs,A
    l=len(label)
    s=np.zeros(A.shape)
    for i in range(l):
        s1=np.zeros(s.shape)
        s2=np.zeros(s.shape)
        for k in range(l):
            xik=rawData[i]-rawData[k]
            s1+=Pij(i,k)*np.dot(xik.transpose(),xik)
        indexOfj=label2indexs[label[i]]
        for j in indexOfj:
            xij=rawData[i]-rawData[j]
            s2+=Pij(i,j)*np.dot(xij.transpose(),xij)
        s+=Pi(i)*s1-s2
    return 2*s*A

def fA():
    global label
    fa=0
    for i in range(len(label)):
        fa+=Pi(i)
    return fa
# main program here
if  __name__ == '__main__':
    pass