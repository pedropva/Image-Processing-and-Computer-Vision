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

#recebe uma imagem e o numero da escala
def amostragem(img,n,filename): 
    amostra = [lin[::n] for lin in img[::n]]
    img = copy.deepcopy(amostra)
    img = np.float32(img)
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return np.array(img)

#recebe uma imagem e o numero de bits de cores
def quantizacao_uniforme(img,K,filename): 
    img = np.float32(img)
    
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            img[i,j] = (math.pow(2,K)-1) * np.float32((img[i,j] - img.min()) / (img.max() - img.min()))
            img[i,j] = np.round(img[i,j]) * int(256/math.pow(2,K))
            
            
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img

#recebe uma imagem e binariza ela
def binarizar(img,filename):
    img = np.float32(img)
    
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            if img[i,j] > 128:
                img[i,j] = true;    
            else:
                img[i,j] = false;
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img
    