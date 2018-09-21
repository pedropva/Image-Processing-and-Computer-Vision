# -*- coding: utf-8 -*-
"""
Feito por Pedropva em 28/08/2018
"""
import numpy as np
import cv2
import os,math, copy
import utils
import random


#translada uma imagem em tx e ty em pixels.
def transladar(img1,tx,ty,filename):
    img = copy.deepcopy(np.float32(img1))

    rows = img.shape[0]
    cols = img.shape[1]
    for i in range(rows):
        for j in range(cols):
            img[i,j] = 0
        
    #aumentando a imagem resultado        
    img = np.concatenate((img,img), axis=0)#dobra a altura 
    img = np.concatenate((img,img), axis=1)#dobra a largura
    img = np.concatenate((img,img), axis=-1)#dobra a largura
    img = np.concatenate((img,img), axis=0)#dobra a altura 
    #resultado final é uma imagem 4*4 vezes maior
    
    #não se engane esse for corre na imagem original
    for i in range(0,rows):
        for j in range(0,cols):
            mat =  [[1, 0,tx],[0, 1,ty],[0, 0, 1]]
            p0 = [] 
            p0.append([i])
            p0.append([j])
            p0.append([1])
            
            p0 = utils.multMatriz(mat,p0)
            #calculada a nova posicao eu coloco a cor no novo lugar
            
            
            #Pego as novas coordenadas e boto elas no centro da nova imagem
            img[int(p0[0][0]+round(1.5*rows)),int(p0[1][0]+round(1.5*cols))] = img1[i,j]
            
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img

#rotaciona uma imagem em volta de um pixel(px,py) com o angulo a em radianos.
def rotacionar(img1,px,py,a,filename):
    img = copy.deepcopy(np.float32(img1))

    rows = img.shape[0]
    cols = img.shape[1]
    for i in range(rows):
        for j in range(cols):
            img[i,j] = 0
        
    #aumentando a imagem resultado        
    img = np.concatenate((img,img), axis=0)#dobra a altura 
    img = np.concatenate((img,img), axis=1)#dobra a largura
    img = np.concatenate((img,img), axis=-1)#dobra a largura
    img = np.concatenate((img,img), axis=0)#dobra a altura 
    #resultado final é uma imagem 4*4 vezes maior
    
    #não se engane esse for corre na imagem original
    for i in range(0,rows):
        for j in range(0,cols):
            matIda =  [[1, 0,-px],[0, 1,-py],[0, 0, 1]]
            mat =  [[math.cos(a), -math.sin(a), 0],[math.sin(a), math.cos(a), 0],[0,0,1]]	
            matVolta =  [[1, 0,px],[0, 1,py],[0, 0, 1]]
            p0 = [] 
            p0.append([i])
            p0.append([j])
            p0.append([1])
            p0 = utils.multMatriz(matIda,p0)#move todos os pixels pro centro em relacao ao pixel dado
            p0 = utils.multMatriz(mat,p0)#rotaciona
            p0 = utils.multMatriz(matVolta,p0)#volta todos os pixels pro lugar
            #calculada a nova posicao eu coloco a cor no novo lugar
            
            
            #Pego as novas coordenadas e boto elas no centro da nova imagem
            img[int(round(p0[0][0])+round(1.5*rows)),int(round(p0[1][0])+round(1.5*cols))] = img1[i,j]
            
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img

#escala uma imagem com tx e ty as taxas de escala em x e y respectivamente.(a partir de 1.alguma coisa, 1 e 1 pra mesma escala)
def escalar(img1,tx,ty,filename):
    img = copy.deepcopy(np.float32(img1))

    rows = img.shape[0]
    cols = img.shape[1]
    for i in range(rows):
        for j in range(cols):
            img[i,j] = 0
        
    #aumentando a imagem resultado        
    img = np.concatenate((img,img), axis=0)#dobra a altura 
    img = np.concatenate((img,img), axis=1)#dobra a largura
    img = np.concatenate((img,img), axis=-1)#dobra a largura
    img = np.concatenate((img,img), axis=0)#dobra a altura 
    #resultado final é uma imagem 4*4 vezes maior
    
    #não se engane esse for corre na imagem original
    for i in range(0,rows):
        for j in range(0,cols):
            
            mat = [[tx,0,0],[0,ty,0],[0,0,1]]
            
            p0 = [] 
            p0.append([i])
            p0.append([j])
            p0.append([1])
            
            p0 = utils.multMatriz(mat,p0)
            
            #calculada a nova posicao eu coloco a cor no novo lugar
            
            
            #Pego as novas coordenadas e boto elas no centro da nova imagem
            img[int(round(p0[0][0])+round(1.5*rows)),int(round(p0[1][0])+round(1.5*cols))] = img1[i,j]
            
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img
