# -*- coding: utf-8 -*-
"""
Feito por Pedropva em 28/08/2018
"""

import numpy as np
import cv2
import os,math, copy
import utils
import random



#soma os pixels de duas imagens de mesmo tamanho
def soma(img1,img2,filename):
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
            if img[i,j] + img2[i,j] > 255: 
                img[i,j] = 255;
            else:
                img[i,j] = img[i,j] + img2[i,j];
                
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img


#soma colorida    
def soma_colorida(img1,img2,filename):

    img = copy.deepcopy(np.float32(img1))
    
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
    
            img[i,j][0] = ((img[i,j][0] + img2[i,j][0]))/2
            img[i,j][1] = ((img[i,j][1] + img2[i,j][1]))/2
            img[i,j][2] = ((img[i,j][2] + img2[i,j][2]))/2
        
                
    if filename != None:
        cv2.imwrite(filename,img) 
    
    #cv2.imwrite("reeeTotal.jpg",imgTotal) 
        
        
    return img

#subtrai os pixels de duas imagens de mesmo tamanho
def subtracao(img1,img2,filename):
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
        
            if img[i,j] - img2[i,j] < 0: 
                img[i,j] = 0;
            else:
                img[i,j] = img[i,j] - img2[i,j];
                
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img

#multiplica os pixels de duas imagens de mesmo tamanho
def multiplicacao(img1,img2,filename):
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
        
            if img[i,j] * img2[i,j] > 255: 
                img[i,j] = 255;
            else:
                if img[i,j] * img2[i,j] < 0:
                    img[i,j] = 0;
                else:
                    img[i,j] = img[i,j] * img2[i,j];
                
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img

#divide os pixels de duas imagens de mesmo tamanho
def divisao(img1,img2,filename):
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
            if img2[i,j] != 0:
                if img[i,j] / img2[i,j] > 255: 
                    img[i,j] = 255;
                else:
                    if img[i,j] / img2[i,j] < 0:
                        img[i,j] = 0;
                    else:
                        img[i,j] = img[i,j] / img2[i,j];
                
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img


#distancia euclidiana entre dois pontos
def distancia_euclidiana(ponto1,ponto2):    
    return math.sqrt(math.pow(2,ponto1[0] - ponto2[0]) + math.pow(2,ponto1[1] - ponto2[1]))


#mistura duas imagens com img1 sendo a primeira imagem, img2 a segunda, peso1 o peso da imagem um na mistura, peso2 opeso da segunda imagem na mistura e alfa sendo um coeficiente somado a mais
def mistura(img1,img2,peso1,peso2,alfa,filename):
    img = np.float32(img1)
    
    if img1.shape[0] > img2 .shape[0]:
        rows = img1.shape[0]
        cols = img1.shape[1]
    else:
        rows = img2.shape[0]
        cols = img2.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            img[i,j] = (img1[i,j]*peso1 + img2[i,j]*peso2)+alfa/(peso1+peso2)
                
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img
