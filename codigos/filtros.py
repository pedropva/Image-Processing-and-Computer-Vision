# -*- coding: utf-8 -*-
"""
Feito por Pedropva em 28/08/2018
"""

import numpy as np
import cv2
import os,math, copy
import utils
import random

#filtro da média, recebe uma imagem e retorna uma imagem de mesmo tamanho, n é o tamanho do kernel
def media(img,n,filename):
    rows = img.shape[0]
    cols = img.shape[1]
    beirada = n//2
    valor = 0
    newImg = np.zeros((rows,cols))
    for i in range(beirada,rows-beirada):
        for j in range(beirada,cols-beirada):
            valor = 0
            for k in range(n):
                for l in range(n):
                    valor +=  img[i-beirada + k,j-beirada + l] #somo todos os valores dos pixels nesse kernel
            newImg[i,j] = round((valor * 1.0)/n**2) # atribui a média da soma desses pixels ao kernel
    if filename != None:
        cv2.imwrite(filename,newImg)
        
    return newImg

#filtro da gaussiano, recebe uma imagem e retorna uma imagem de mesmo tamanho com a filtragem gaussiana (passabaixa/desfoque)
def gaussiano(img,filename):
    #fazendo o kernel
    kernel =[[1,2,1],[2,4,2],[1,2,1]]
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
                    
            newImg[i,j] = round((valor * 1.0)/16)# atribui a média da soma desses pixels ao kernel
    if filename != None:
        cv2.imwrite(filename,newImg)
        
    return newImg

#filtro da mediana, recebe uma imagem e retorna uma imagem de mesmo tamanho com a filtragem gaussiana (passabaixa/desfoque)
def mediana(img,filename):
    rows = img.shape[0]
    cols = img.shape[1]
    beirada = 3//2 #3 eh o numero delinhas da matriz
    newImg = np.zeros((rows,cols))
    for i in range(rows-beirada):
        for j in range(cols-beirada):
            vizinhos = [] #variavel que guarda a lista dos pixels da janela equivalente na imagem
            for k in range(3):
                for l in range(3):
                    vizinhos.append(img[i-beirada + k,j-beirada + l]) #multiplico os valores do kernel pela janela equivalente dos pixels                    
            #print(vizinhos)
            #print(np.median(vizinhos))
            newImg[i,j] = np.median(vizinhos) # atribui a mediana desses pixels ao pixel
    if filename != None:
        cv2.imwrite(filename,newImg)
        
    return newImg