# -*- coding: utf-8 -*-
"""
Feito por Pedropva em 28/08/2018
"""

import numpy as np
import cv2
import os,math, copy
import utils
import random


#dithering, recebe uma imagem em escala de cinza e retorna uma em preto e branco com baixa qualidade
def basicDithering(img,filename):
    img = np.float32(img)

    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            if img[i,j] > 127:
                img[i,j] = 255    
            else:
                img[i,j] = 0
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img
    

#dithering, recebe uma imagem em escala de cinza e retorna uma em preto e branco com baixa qualidade e com pixels randomizados
def randomDithering(img,filename):
    img = np.float32(img)
    
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            if img[i,j] + random.randint(-127,127) > 127:
                img[i,j] = 255;    
            else:
                img[i,j] = 0;
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img

#dithering, recebe uma imagem em escala de cinza e retorna uma em preto e branco baseado em Algoritmo Ordenado Periodico com pixels Aglomerados
def AlgoritmoOrdenadoPeriodicoAglomerado(img,filename):
    img = np.float32(img)
    ditheringMatrix = [[8,3,4],[6,1,2],[7,5,9]]
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            temp1 = (img[i,j]* 1.0)/255
            temp2 = (ditheringMatrix[i%3][j%3]* 1.0)/9
            if temp1 > temp2 :
                img[i,j] = 255;
            else:
                img[i,j] = 0;
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return np.uint8(img)

#dithering, recebe uma imagem em escala de cinza e retorna uma em preto e branco baseado em Algoritmo Ordenado Periodico com pixels Dispersos
def AlgoritmoOrdenadoPeriodicoDisperso(img,filename):
    img = np.float32(img)
    ditheringMatrix = [[2,3],[4,1]]
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            temp1 = (img[i,j]* 1.0)/255
            temp2 = (ditheringMatrix[i%2][j%2]* 1.0)/5
            if temp1 > temp2 :
                img[i,j] = 255;
            else:
                img[i,j] = 0;
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return np.uint8(img)

#dithering, recebe uma imagem em escala de cinza e retorna uma em preto e branco baseado em Algoritmo aperiódico (Floyd-Steinberg).
def AlgoritmoAperiodico(img,filename):
    img = np.float32(img)
    
    rows = img.shape[0]
    cols = img.shape[1]
    copy = 0
    for i in range(rows):
        for j in range(cols):
            copy = img[i,j]
            if img[i,j] > 127:
                img[i,j] = 255;    
            else:
                img[i,j] = 0;
            
            
            erro = copy  -  img[i,j]  
            if(i+1 < rows):
                img[i+1,j] = img[i+1,j] + erro * 7.0/16 
            if(i+1 < rows and j+1 < cols):
                img[i+1,j+1] = img[i+1,j+1] + erro * 1.0/16 
            if(j+1 < cols):
                img[i,j+1] = img[i,j+1] + erro * 5.0/16    
            if(i-1 > 0 and j+1 < cols):
                img[i-1,j+1] = img[i-1,j+1] + erro * 3.0/16
    if filename != None:
        cv2.imwrite(filename,img)
    return np.uint8(img)
