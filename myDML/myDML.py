# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 24:00:00 2017

@author: Zhiyu
"""
# import your module here
import numpy as np
from NCA import NCA
from threading import Thread
import functools
import math
from bdgNCA import bdgNCA
# (global) variable definition here
TRAINING_TIME_LIMIT = 60*10
global nca,bdgnca
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
    global nca,bdgnca
    #nca=NCA(traindata)
    #nca.train()
    bdgnca=bdgNCA(traindata)
    bdgnca.train()

def Euclidean_distance(inst_a, inst_b):
    return np.linalg.norm(inst_a - inst_b)

def distance(inst_a, inst_b):           #行向量
    return bdgnca.myDistance(inst_a,inst_b)   #(A(x-y))T(A(x-y))
# main program here
if  __name__ == '__main__':
    pass