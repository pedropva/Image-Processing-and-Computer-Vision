# -*- coding: utf-8 -*-
"""
Feito por Pedropva em 28/08/2018
"""

import numpy as np
import cv2
import os,math, copy
import utils
import random

#realce negativo recebe umas imagem e retorna ela negativada
def negacao(img,filename):
    
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
                img[i,j] = 255 - img[i,j];
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img
    
#realce contraste recebe umas imagem e retorna ela com um novo range de contraste, indo de minC a maxC
def contraste(img,minC,maxC,filename):
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
                img[i,j] = (img[i,j] - np.amin(img))*((maxC - minC)/(np.amax(img) - np.amin(img))) + minC
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img

#realce gama recebe umas imagem e retorna ela com gama mudado
#Fator gama γ > 1: comprime as intensidades de preto (regiões escuras),enquanto expande as intensidades claras.
#Fator gama 0 < γ < 1: operação inversa.
def gama(img,c,gama,filename):
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
                aux = (img[i,j]*1.0)/255
                value = c * (aux)**gama
                value = value*255
                if(value > 255):
                    img[i,j] = 255
                else:
                    img[i,j] = value
                
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img

#realce linear, recebe uma imagem, retorna ela com o contraste maior ou menor, sendo G o ganho desejado e D o fator de incremento
#Aumenta o contraste da imagem, expandindo o intervalo original de níveis de cinza
def linear(img,G,D,filename):
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
                value = G*img[i,j] + D
                if(value > 255):
                    img[i,j] = 255
                else:
                    img[i,j] = value
                
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img


#realce logaritmico, recebe uma imagem, retorna ela com o contraste maior ou menor, sendo G o fator definido entre 0 e 255
#Aumenta o contraste em regiões escuras (valores de cinza baixos). Equivale a uma curva logarítmica.
def logaritmico(img,filename):
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
                G = 255.0/np.log10(255)
                value = G*np.log10(img[i,j] + 1)
                if(value > 255):
                    img[i,j] = 255
                else:
                    img[i,j] = value
                
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img


#realce quadratico, recebe uma imagem, retorna ela com o contraste maior ou menor
#Aumenta o contraste em regiões claras (valores de cinza altos).
def quadratico(img,filename):
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
                G = 1.0/255
                value = G*((img[i,j])**2)
                if(value > 255):
                    img[i,j] = 255
                else:
                    img[i,j] = value
                
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img

#realce da raiz quadrada, recebe uma imagem, retorna ela com o contraste maior ou menor
#Aumenta o contraste das regiões escuras da imagem original.
#Difere do logarítmico porque realça um intervalo maior de níveis de cinza baixos.
def raiz(img,filename):
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
                G = 255/np.sqrt(255)
                img[i,j] = G*(np.sqrt(img[i,j]))
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img