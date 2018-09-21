# -*- coding: utf-8 -*-
"""
Feito por Pedropva em 28/08/2018
"""

import numpy as np
import cv2
import os,math, copy
import utils
import random

#histograma basico    
def histograma(img):
    rows = img.shape[0]
    cols = img.shape[1]
    valores = [0] * 256
    for i in range(rows):
        for j in range(cols):
                valores[img[i,j]] += 1          
        
    return valores

#acumula os valores do histograma
def cumulativo(img):
    rows = img.shape[0]
    cols = img.shape[1]
    valores = [0] * 256
    for i in range(rows):
        for j in range(cols):
                valores[img[i,j]] += 1          
        
    for i in range(1,256):
                valores[i] += valores[i-1]
        
    return valores

#equalização do histograma
def equalizado(img,filename):
    newImg = copy.deepcopy(img)
    rows = img.shape[0]
    cols = img.shape[1]
    fator = 255.0/ (rows * cols)
    valores = [0] * 256 #lista com os valores do histograma equalizado
    for i in range(rows):
        for j in range(cols):
                valores[img[i,j]] += 1          
                
    #faz o histograma cumulativo
    for i in range(1,256):
                valores[i] += valores[i-1]                
                
    #multiplica o resultado pelo fator
    for i in range(1,256):
                valores[i] = round(valores[i]*1.0*fator)            
                
    #mapeia o histograma equalizado pra imagem
    for i in range(rows):
        for j in range(cols):
            newImg[i,j] = valores[img[i,j]]
    
    if filename != None:
        cv2.imwrite(filename,newImg) 
    
    return histograma(newImg) #retorna ohistograma da nova imagem equalizada
    
#histograma alongado
def alongado(img,plow,phigh,filename):
    rows = img.shape[0]
    cols = img.shape[1]
    newImg = copy.deepcopy(img)
    for i in range(rows):
        for j in range(cols):
            if(img[i,j] <= plow):
                newImg[i,j]= 0
            elif(img[i,j] >= phigh):
                newImg[i,j] = 255
            else:
                newImg[i,j] = round(255 * (img[i,j] - plow)/(phigh - plow))
            
    if filename != None:
        cv2.imwrite(filename,newImg)
        
    return newImg

    
#histograma especificado
def especificado(img,img2,filename):
    newImg = copy.deepcopy(img)
    histEq1 = equalizado(img,None)
    histEq2 = equalizado(img2,None)
    histEspecificado = []
    menor_indice = 0
    for i in range(len(histEq1)):
        for j in range(len(histEq2)):
            if abs(histEq1[i] - histEq2[j]) < abs(histEq1[i]-histEq1[menor_indice]):
                menor_indice = j
        histEspecificado.append(menor_indice)
        menor_indice = 0
    for i in range(256):
        for j in range(256):
            newImg[i,j] = histEspecificado[img[i,j]]
            
    if filename != None:
        cv2.imwrite(filename,newImg)
        
    return histograma(newImg)