
# -*- coding: utf-8 -*-
"""
Feito por Pedropva em 28/08/2018
"""

import numpy as np
import cv2
import os,math, copy
import utils
import random


def separar_canais(img,filenameBlue,filenameGreen,filenameRed):
    imgBlue = copy.deepcopy(img)
    imgGreen = copy.deepcopy(img)
    imgRed = copy.deepcopy(img)   
    
    imgTotal = copy.deepcopy(img)   
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            imgRed[i,j][1] = 0
            imgRed[i,j][0] = 0
            
            imgGreen[i,j][2] = 0
            imgGreen[i,j][0] = 0
            
            
            imgBlue[i,j][1] = 0
            imgBlue[i,j][2] = 0
            
            imgTotal[i,j][0] = imgBlue[i,j][0] + imgRed[i,j][0] + imgGreen[i,j][0]
            imgTotal[i,j][1] = imgBlue[i,j][1] + imgRed[i,j][1] + imgGreen[i,j][1]
            imgTotal[i,j][2] = imgBlue[i,j][2] + imgRed[i,j][2] + imgGreen[i,j][2]
            #print(img [i,j])

    if filenameBlue != None:
        cv2.imwrite(filenameBlue,imgBlue)
    if filenameGreen != None:
        cv2.imwrite(filenameGreen,imgGreen)
    if filenameRed != None:
        cv2.imwrite(filenameRed,imgRed) 
    
    #cv2.imwrite("reeeTotal.jpg",imgTotal) 
        
        
    return img

#B.G.R. TO C.M.Y. recebe uma imagem em bgr e retorna uma imagem em cmy
    
def bgrToCmy(img1,filename):

    img = copy.deepcopy(img1)
     
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            img[i,j][0] = 255 - img1[i,j][0]
            img[i,j][1] = 255 - img1[i,j][1]
            img[i,j][2] = 255 - img1[i,j][2]

    if filename != None:
        cv2.imwrite(filename,img) 
    
    #cv2.imwrite("reeeTotal.jpg",imgTotal) 
        
        
    return img

#B.G.R. TO Y.U.V. recebe uma imagem em bgr e retorna uma imagem em yuv
    
def bgrToYuv(img1,filename):
    img1 = np.float32(img1)
    img = copy.deepcopy(img1)
     
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            img[i,j][0] = 0.299*img1[i,j][2] + 0.587*img1[i,j][1] + 0.144*img1[i,j][0]     #Y =  0.299*R + 0.587*G + 0.144*B
            img[i,j][1] = img1[i,j][0] - img[i,j][0]    #U = B- Y
            img[i,j][2] = img1[i,j][2] - img[i,j][0]    #V = R - Y

    if filename != None:
        cv2.imwrite(filename,img) 
    
    #cv2.imwrite("reeeTotal.jpg",imgTotal) 
        
        
    return img

#B.G.R. TO Y.Cr.Cb. recebe uma imagem em bgr e retorna uma imagem em ycrcb
#delta é 128 pra imagens 8bits,32568 pra 16 bits e 0.5 pra floats
def bgrToYCrCb(img1,delta,filename):
    img1 = np.float32(img1)
    img = copy.deepcopy(img1)
     
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            img[i,j][0] = 0.299*img1[i,j][2] + 0.587*img1[i,j][1] + 0.114*img1[i,j][0]    #Y = 0.299*R + 0.587*G + 0.114*B
            img[i,j][1] = (img1[i,j][2] - img[i,j][0])* 0.713 + delta    #Cr = (R - Y)* 0.713 + delta 
            img[i,j][2] = (img1[i,j][0] - img[i,j][0])* 0.564 + delta    #Cb = (B - Y)* 0.564 + delta

    if filename != None:
        cv2.imwrite(filename,img) 
        
        
    return img



#B.G.R. TO Y.I.Q. recebe uma imagem em bgr e retorna uma imagem em yiq
def bgrToYiQ(img1,filename):

    img = copy.deepcopy(img1)
     
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            
            mat = [[0.299,0.587,0.144],[0.596,-0.275,-0.321],[0.212,-0.523,0.311]]
            
            p0 = [] 
            p0.append([img1[i,j][2]])
            p0.append([img1[i,j][1]])
            p0.append([img1[i,j][0]])
            
            p0 = utils.multMatriz(mat,p0)
            
            #calculada a nova posicao eu coloco a cor no novo lugar
            
            
            #Pego as novas coordenadas e boto elas no centro da nova imagem
            img[i,j][0] = p0[0][0] #Y
            img[i,j][1] = p0[1][0] #I
            img[i,j][2] = p0[2][0] #Q

    if filename != None:
        cv2.imwrite(filename,img) 
           
    return img


#B.G.R. TO R.G.B. recebe uma imagem em bgr e retorna uma imagem em rgb
def bgrTorgb(img1,filename):

    img = copy.deepcopy(img1)
     
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            img[i,j][0] = img1[i,j][2] #R
            img[i,j][1] = img1[i,j][1] #G
            img[i,j][2] = img1[i,j][0] #B

    if filename != None:
        cv2.imwrite(filename,img) 
           
    return img
