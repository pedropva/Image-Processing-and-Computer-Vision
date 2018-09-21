# -*- coding: utf-8 -*-
"""
Feito por Pedropva em 28/08/2018
"""

import numpy as np
import cv2
import os,math, copy
import utils
import random


#detecção de pontos isolados
def pontos(img,limiar,filename):
    #fazendo o kernel
    kernel =[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    rows = img.shape[0]
    cols = img.shape[1]
    beirada = 3//2 #3 eh o numero delinhas da matriz
    valor = 0
    i = beirada
    j = beirada
    for i in range(rows-beirada):
        for j in range(cols-beirada):
            valor = 0
            for k in range(3):
                for l in range(3):
                    valor +=  img[i-beirada + k,j-beirada + l] * kernel[k][l] #multiplico os valores do kernel pela janela equivalente dos pixels
                    
            img[i,j] = valor# atribui a média da soma desses pixels ao kernel
            
    for i in range(rows):
        for j in range(cols):
            if img[i,j] <= limiar:
                img[i,j] = 0
            else: 
                img[i,j] = 255
            
            
            
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img