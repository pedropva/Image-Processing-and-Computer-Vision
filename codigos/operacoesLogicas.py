# -*- coding: utf-8 -*-
"""
Feito por Pedropva em 28/08/2018
"""

import numpy as np
import cv2
import os,math, copy
import utils
import random



true = 255
false = 0

#recebe duas imagens binarizadas e faz o AND bitwise
def And(img1,img2,filename): 
    img = np.float32(img1)
    
    if img1.shape[0] < img2 .shape[0]:
        rows = img1.shape[0]
    else:
        rows = img2.shape[0]
    
    if img1.shape[1] < img2 .shape[1]:
        cols = img1.shape[1]
    else:
        cols = img2.shape[1]
    
    
    for i in range(rows):
        for j in range(cols):
            if img1[i,j] == true and img2[i,j] == true:
                img[i,j] = true;    
            else:
                img[i,j] = false;
                
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img

#recebe duas imagens binarizadas e faz o OR bitwise
def Or(img1,img2,filename): 
    img = np.float32(img1)
    
    if img1.shape[0] < img2 .shape[0]:
        rows = img1.shape[0]
    else:
        rows = img2.shape[0]
    
    if img1.shape[1] < img2 .shape[1]:
        cols = img1.shape[1]
    else:
        cols = img2.shape[1]
    
    
    for i in range(rows):
        for j in range(cols):
            if img1[i,j] == true or img2[i,j] == true:
                img[i,j] = true;    
            else:
                img[i,j] = false;
                
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img

#recebe duas imagens binarizadas e faz o XOR bitwise
def Xor(img1,img2,filename): 
    img = np.float32(img1)
    
    if img1.shape[0] < img2 .shape[0]:
        rows = img1.shape[0]
    else:
        rows = img2.shape[0]
    
    if img1.shape[1] < img2 .shape[1]:
        cols = img1.shape[1]
    else:
        cols = img2.shape[1]
    
    
    for i in range(rows):
        for j in range(cols):
            if img1[i,j] != img2[i,j]:
                img[i,j] = true;    
            else:
                img[i,j] = false;
                
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img


#recebe uma imagem binarizada e faz o NOT bitwise
def Not(img,filename):
    img = np.float32(img)
    
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            if img[i,j] == true:
                img[i,j] = false;    
            else:
                img[i,j] = true;
                
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img