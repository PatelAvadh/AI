# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:35:45 2020

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

def do_nestrov_accelerted_gradient_descent():
    
    tic = time.time()
    w,b,eta,max_epoch = 0,0,1.0,10
    prev_v_w,prev_v_b, gamma =0,0,0.1
    for i in range(max_epoch):
        dw,db = 0,0
        
        v_w = gamma* prev_v_w
        v_b = gamma* prev_v_b
        for x,y in zip(X,Y):
            dw += grad_w(w-v_w, b-v_b, x, y)
            db += grad_b(w-v_w, b-v_b, x, y)
            
        v_w = gamma* prev_v_w + eta*dw
        v_b = gamma* prev_v_b + eta*db
        
        w = w - v_w
        b = b - v_b
        
        prev_v_w = v_w
        prev_v_b = v_b
      
        err = error(w,b)
        toc = time.time()
    print("Final Weight :",w)
    print("-------------")
    print("Final Bias : ",b)
    print("-------------")
    print("Nestrov Gradient Error",err)
    print("Time of execution is: ",toc-tic)    
do_nestrov_accelerted_gradient_descent()
