# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:19:26 2020

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

def do_gradient_descent():
    tic = time.time()
    w,b,lr,max_epoch = 0.9,0.01,17,7000
    print(f"Weight {w}, Bias {b}, Learning_rate {lr}, Epoch {max_epoch}")
    print("--------------")
    for i in range(max_epoch):
        dw =0
        db =0
        for x,y in zip(X,Y):
            dw = dw + grad_w(w, b, x, y)
            db = db + grad_b(w, b, x, y)
        w = w - (lr*dw)
        b = b - (lr*db)
    toc = time.time()
    print("Final Weight :",w)
    print("-------------")
    print("Final Bias : ",b)
    print("-------------")
    err = error(w,b)
    print("Gradient descent Error",err)    
    print("Time of execution is: ",toc-tic)
do_gradient_descent()

