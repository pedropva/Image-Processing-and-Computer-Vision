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
    newImg = np.zeros((rows,cols))
    for i in range(beirada,rows-beirada):
        for j in range(beirada,cols-beirada):
            valor = 0
            for k in range(3):
                for l in range(3):
                    valor +=  img[i-beirada + k,j-beirada + l] * kernel[k][l] #multiplico os valores do kernel pela janela equivalente dos pixels
            
            if valor <= limiar:
                newImg[i,j] = 0
            else: 
                newImg[i,j] = 255      
            
    if filename != None:
        cv2.imwrite(filename,newImg)
        
    return newImg

#detecção de retas
def retas(img,angulo,limiar,filename):
    #fazendo o kernel
    if angulo == 0:
        kernel =[[-1,-1,-1],[2,2,2],[-1,-1,-1]]    
    elif angulo == 45:
        kernel =[[-1,-1,2],[-1,2,-1],[2,-1,-1]]    
    elif angulo == 90:
        kernel =[[-1,2,-1],[-1,2,-1],[-1,2,-1]]    
    elif angulo == -45:
        kernel =[[2,-1,-1],[-1,2,-1],[-1,-1,2]]    
    else:
        kernel =[[-1,-1,-1],[2,2,2],[-1,-1,-1]]    

    rows = img.shape[0]
    cols = img.shape[1]
    beirada = 3//2 #3 eh o numero delinhas da matriz
    valor = 0
    newImg = np.zeros((rows,cols))
    for i in range(beirada,rows-beirada):
        for j in range(beirada,cols-beirada):
            valor = 0
            for k in range(3):
                for l in range(3):
                    valor +=  img[i-beirada + k,j-beirada + l] * kernel[k][l] #multiplico os valores do kernel pela janela equivalente dos pixels            
            if abs(valor) <= limiar:
                newImg[i,j] = 0
            else: 
                newImg[i,j] = 255      
            
    if filename != None:
        cv2.imwrite(filename,newImg)
        
    return newImg

#detecção de bordas roberts
def roberts(img,limiar,filename):
    #fazendo o kernel
    kernelX =[[1,0],[0,-1]]
    kernelY =[[0,-1],[1,0]]

    rows = img.shape[0]
    cols = img.shape[1]
    beirada = 2//2 #2 eh o numero de linhas da matriz
    valor = 0
    imgX = np.zeros((rows,cols))
    imgResult = np.zeros((rows,cols))

    for i in range(beirada,rows-beirada):
        for j in range(beirada,cols-beirada):
            valor = 0
            for k in range(2):
                for l in range(2):
                    valor +=  img[i-beirada + k,j-beirada + l] * kernelX[k][l] #multiplico os valores do kernel pela janela equivalente dos pixels            
            imgX[i,j] = valor
    #fim das convolucoes no eixo x, agora usa a saida disso pra fazer as convolucoes em y
    for i in range(beirada,rows-beirada):
        for j in range(beirada,cols-beirada):
            valor = 0
            for k in range(2):
                for l in range(2):
                    valor +=  img[i-beirada + k,j-beirada + l] * kernelY[k][l] #multiplico os valores do kernel pela janela equivalente dos pixels            
            #valor guarda convolução nesse pixel em y e imgX guarda o dado desse pixel na convolução em X
            if abs(valor) + abs(imgX[i,j]) <= limiar:
                imgResult[i,j] = 0
            else: 
                imgResult[i,j] = 255      
            
    if filename != None:
        cv2.imwrite(filename,imgResult)
        
    return imgResult


#detecção de bordas prewitt
def prewitt(img,limiar,filename):
    #fazendo o kernel
    kernelX = [[-1,-1,-1],[0,0,0],[1,1,1]]  
    kernelY =[[-1,0,1],[-1,0,1],[-1,0,1]]  

    rows = img.shape[0]
    cols = img.shape[1]
    beirada = 3//2 #2 eh o numero de linhas da matriz
    valor = 0
    imgX = np.zeros((rows,cols))
    imgResult = np.zeros((rows,cols))

    for i in range(beirada,rows-beirada):
        for j in range(beirada,cols-beirada):
            valor = 0
            for k in range(3):
                for l in range(3):
                    valor +=  img[i-beirada + k,j-beirada + l] * kernelX[k][l] #multiplico os valores do kernel pela janela equivalente dos pixels            
            imgX[i,j] = valor
    #fim das convolucoes no eixo x, agora usa a saida disso pra fazer as convolucoes em y
    for i in range(beirada,rows-beirada):
        for j in range(beirada,cols-beirada):
            valor = 0
            for k in range(3):
                for l in range(3):
                    valor +=  img[i-beirada + k,j-beirada + l] * kernelY[k][l] #multiplico os valores do kernel pela janela equivalente dos pixels            
            #valor guarda convolução nesse pixel em y e imgX guarda o dado desse pixel na convolução em X
            if abs(valor) + abs(imgX[i,j]) <= limiar:
                imgResult[i,j] = 0
            else: 
                imgResult[i,j] = 255      
            
    if filename != None:
        cv2.imwrite(filename,imgResult)
        
    return imgResult


#detecção de bordas sobel
def sobel(img,limiar,filename):
    #fazendo o kernel
    kernelX = [[-1,0,1],[-2,0,2],[-1,0,1]]  
    kernelY =[[-1,-2,-1],[0,0,0],[1,2,1]]  

    rows = img.shape[0]
    cols = img.shape[1]
    beirada = 3//2 #2 eh o numero de linhas da matriz
    valor = 0
    imgX = np.zeros((rows,cols))
    imgResult = np.zeros((rows,cols))

    for i in range(beirada,rows-beirada):
        for j in range(beirada,cols-beirada):
            valor = 0
            for k in range(3):
                for l in range(3):
                    valor +=  img[i-beirada + k,j-beirada + l] * kernelX[k][l] #multiplico os valores do kernel pela janela equivalente dos pixels            
            imgX[i,j] = valor
    #fim das convolucoes no eixo x, agora usa a saida disso pra fazer as convolucoes em y
    for i in range(beirada,rows-beirada):
        for j in range(beirada,cols-beirada):
            valor = 0
            for k in range(3):
                for l in range(3):
                    valor +=  img[i-beirada + k,j-beirada + l] * kernelY[k][l #multiplico os valores do kernel pela janela equivalente dos pixels            
            #valor guarda convolução nesse pixel em y e imgX guarda o dado desse pixel na convolução em X
            if abs(valor) + abs(imgX[i,j]) <= limiar:
                imgResult[i,j] = 0
            else: 
                imgResult[i,j] = 255      
    if filename != None:
        cv2.imwrite(filename,imgResult)
        
    return imgResult


#detecção de bordas laplaciano
def laplaciano(img,limiar,filename):
    #fazendo o kernel
    kernelX = [[0,1,0],[1,-4,1],[0,1,0]]  
    kernelY =[[2,-1,2],[-1,-4,-1],[2,-1,2]]  

    rows = img.shape[0]
    cols = img.shape[1]
    beirada = 3//2 #2 eh o numero de linhas da matriz
    valor = 0
    imgX = np.zeros((rows,cols))
    imgResult = np.zeros((rows,cols))

    for i in range(beirada,rows-beirada):
        for j in range(beirada,cols-beirada):
            valor = 0
            for k in range(3):
                for l in range(3):
                    valor +=  img[i-beirada + k,j-beirada + l] * kernelX[k][l] #multiplico os valores do kernel pela janela equivalente dos pixels            
            imgX[i,j] = valor
    #fim das convolucoes no eixo x, agora usa a saida disso pra fazer as convolucoes em y
    for i in range(beirada,rows-beirada):
        for j in range(beirada,cols-beirada):
            valor = 0
            for k in range(3):
                for l in range(3):
                    valor +=  img[i-beirada + k,j-beirada + l] * kernelY[k][l] #multiplico os valores do kernel pela janela equivalente dos pixels            
            #valor guarda convolução nesse pixel em y e imgX guarda o dado desse pixel na convolução em X
            if abs(valor) + abs(imgX[i,j]) <= limiar:
                imgResult[i,j] = 0
            else: 
                imgResult[i,j] = 255      
            
    if filename != None:
        cv2.imwrite(filename,imgResult)
        
    return imgResult
