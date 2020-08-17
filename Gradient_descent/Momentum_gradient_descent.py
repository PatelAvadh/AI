# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:26:15 2020

@author: Dell
"""
import time
import numpy as np
X = [0.5,2.5]
Y = [0.2,0.9]

def f(w,b,x):
    return 1/(1+np.exp(-(w*x+b)))

def error(w,b):
    err = 0.0
    for x,y in zip(X,Y):
        fx = f(w,b,x)
        err = err + 0.5*(fx-y)**2
        return err

def grad_w(w,b,x,y):
    fx = f(w,b,x)
    return (fx-y)*x*(1-fx)*fx

def grad_b(w,b,x,y):
    fx = f(w,b,x)
    return (fx-y)*(1-fx)*fx

def do_momentum_gradient_descent():
    tic = time.time()
    w,b,eta,max_epoch = 0,0,1,10
    gamma = 0.1
    prev_v_w,prev_v_b = 0,0
    for i in range(max_epoch):
        dw,db = 0,0
        for (x,y) in zip(X,Y):
            dw += grad_w(w,b,x,y)
            db += grad_b(w,b,x,y)
        prev_v_w = w = w - ((gamma * prev_v_w) +(eta * dw))
        prev_v_b = b - ((gamma * prev_v_b) +(eta * db))
        
    toc = time.time()
    
    print("Final Weight :",w)
    print("-------------")
    print("Final Bias : ",b)
    print("-------------")
    print("Momentum Gradient Descent : ",error(w,b))
    print("Time of execution is: ",toc-tic)

do_momentum_gradient_descent()
