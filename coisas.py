# -*- coding: utf-8 -*-
"""
Feito por Pedropva em 28/08/2018, meu aniversário :p
"""

import numpy as np
import cv2
import os,math, copy
import utils
import random
import matplotlib.pyplot as plt

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
    



#recebe duas imagens binarizadas e faz o AND bitwise
def operacoes_logicas_and(img1,img2,filename): 
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
            if img1[i,j] == true and img2[i,j] == true:
                img[i,j] = true;    
            else:
                img[i,j] = false;
                
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img

#recebe duas imagens binarizadas e faz o OR bitwise
def operacoes_logicas_or(img1,img2,filename): 
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
            if img1[i,j] == true or img2[i,j] == true:
                img[i,j] = true;    
            else:
                img[i,j] = false;
                
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img

#recebe duas imagens binarizadas e faz o XOR bitwise
def operacoes_logicas_xor(img1,img2,filename): 
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
            if img1[i,j] != img2[i,j]:
                img[i,j] = true;    
            else:
                img[i,j] = false;
                
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img


#recebe uma imagem binarizada e faz o NOT bitwise
def operacoes_logicas_not(img,filename):
    img = np.float32(img)
    
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            if img[i,j] == true:
                img[i,j] = false;    
            else:
                img[i,j] = true;
                
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img

#soma os pixels de duas imagens de mesmo tamanho
def operacoes_aritmeticas_soma(img1,img2,filename):
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

#subtrai os pixels de duas imagens de mesmo tamanho
def operacoes_aritmeticas_subtracao(img1,img2,filename):
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
def operacoes_aritmeticas_multiplicacao(img1,img2,filename):
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
def operacoes_aritmeticas_divisao(img1,img2,filename):
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
            img[p0[0][0]+round(1.5*rows),p0[1][0]+round(1.5*cols)] = img1[i,j]
            
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
            img[round(p0[0][0])+round(1.5*rows),round(p0[1][0])+round(1.5*cols)] = img1[i,j]
            
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
            img[round(p0[0][0])+round(1.5*rows),round(p0[1][0])+round(1.5*cols)] = img1[i,j]
            
    if filename != None:    
        cv2.imwrite(filename,img)
            
    return img

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
            img[i,j][0] = p0[0] #Y
            img[i,j][1] = p0[1] #I
            img[i,j][2] = p0[2] #Q

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


#dithering, recebe uma imagem em escala de cinza e retorna uma em preto e branco com baixa qualidade
def basicDithering(img,filename):
    img = np.float32(img)
    
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            if img[i,j] > 127:
                img[i,j] = true;    
            else:
                img[i,j] = false;
    
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
                img[i,j] = true;    
            else:
                img[i,j] = false;
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img

#dithering, recebe uma imagem em escala de cinza e retorna uma em preto e branco baseado em Algoritmo Ordenado Periodico com pixels Aglomerados
def AlgoritmoOrdenadoPeriodicoAglomeradoDithering(img,filename):
    img = np.float32(img)
    ditheringMatrix = [[8,3,4],[6,1,2],[7,5,9]]
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            temp1 = (img[i,j]* 1.0)/true
            temp2 = (ditheringMatrix[i%3][j%3]* 1.0)/9
            if temp1 > temp2 :
                img[i,j] = true;
            else:
                img[i,j] = false;
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return np.uint8(img)

#dithering, recebe uma imagem em escala de cinza e retorna uma em preto e branco baseado em Algoritmo Ordenado Periodico com pixels Dispersos
def AlgoritmoOrdenadoPeriodicoDispersoDithering(img,filename):
    img = np.float32(img)
    ditheringMatrix = [[2,3],[4,1]]
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            temp1 = (img[i,j]* 1.0)/true
            temp2 = (ditheringMatrix[i%2][j%2]* 1.0)/5
            if temp1 > temp2 :
                img[i,j] = true;
            else:
                img[i,j] = false;
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return np.uint8(img)

#dithering, recebe uma imagem em escala de cinza e retorna uma em preto e branco baseado em Algoritmo aperiódico (Floyd-Steinberg).
def AlgoritmoAperiodicoDithering(img,filename):
    img = np.float32(img)
    
    rows = img.shape[0]
    cols = img.shape[1]
    copy = 0
    for i in range(rows):
        for j in range(cols):
            copy = img[i,j]
            if img[i,j] > 127:
                img[i,j] = true;    
            else:
                img[i,j] = false;
            
            
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

#filtro da média, recebe uma imagem e retorna uma imagem de mesmo tamanho, n é o tamanho do kernel
def filtro_media(img,n,filename):
    rows = img.shape[0]
    cols = img.shape[1]
    beirada = n//2
    valor = 0

    i = beirada
    j = beirada
    for i in range(rows-beirada):
        for j in range(cols-beirada):
            valor = 0
            for k in range(n):
                for l in range(n):
                    valor +=  img[i-beirada + k,j-beirada + l] #somo todos os valores dos pixels nesse kernel
            img[i,j] = round((valor * 1.0)/n**2) # atribui a média da soma desses pixels ao kernel
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img

#filtro da gaussiano, recebe uma imagem e retorna uma imagem de mesmo tamanho com a filtragem gaussiana (passabaixa/desfoque)
def filtro_gaussiano(img,filename):
    #fazendo o kernel
    kernel =[[1,2,1],[2,4,2],[1,2,1]]
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
                    
            img[i,j] = round((valor * 1.0)/16)# atribui a média da soma desses pixels ao kernel
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img

#filtro da mediana, recebe uma imagem e retorna uma imagem de mesmo tamanho com a filtragem gaussiana (passabaixa/desfoque)
def filtro_mediana(img,filename):
    rows = img.shape[0]
    cols = img.shape[1]
    beirada = 3//2 #3 eh o numero delinhas da matriz
    i = beirada
    j = beirada
    
    for i in range(rows-beirada):
        for j in range(cols-beirada):
            vizinhos = [] #variavel que guarda a lista dos pixels da janela equivalente na imagem
            for k in range(3):
                for l in range(3):
                    vizinhos.append(img[i-beirada + k,j-beirada + l]) #multiplico os valores do kernel pela janela equivalente dos pixels                    
            #print(vizinhos)
            #print(np.median(vizinhos))
            img[i,j] = np.median(vizinhos) # atribui a mediana desses pixels ao pixel
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img



#realce negativo recebe umas imagem e retorna ela negativada
def realce_negacao(img,filename):
    
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
                img[i,j] = 255 - img[i,j];
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img
    
#realce contraste recebe umas imagem e retorna ela com um novo range de contraste, indo de minC a maxC
def realce_contraste(img,minC,maxC,filename):
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
def realce_gama(img,c,gama,filename):
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
def realce_linear(img,G,D,filename):
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
def realce_logaritmico(img,filename):
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
def realce_quadratico(img,filename):
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
def realce_raiz(img,filename):
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
                G = 255/np.sqrt(255)
                img[i,j] = G*(np.sqrt(img[i,j]))
    
    if filename != None:
        cv2.imwrite(filename,img)
        
    return img


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
def histograma_cumulativo(valores):
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
def histograma_equalizado(img,filename):
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
def histograma_alongado(img,plow,phigh,filename):
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
def histograma_especificado(img,img2,filename):
    newImg = copy.deepcopy(img)
    histEq1 = histograma_equalizado(img,None)
    histEq2 = histograma_equalizado(img2,None)
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


#detecção de pontos isolados
def deteccao_pontos(img,limiar,filename):
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

















































if __name__ == "__main__":
    
    filename = 'reee.jpg'
    img = cv2.imread(filename,0)
    name, extension = os.path.splitext(filename)
    
    filename2 = 'reee.jpg'
    img2 = cv2.imread(filename2,0)
    name2, extension2 = os.path.splitext(filename2)
    
    #aqui eu faço amostragem de 2 e 4 e salvo as duas imagens
    '''
    fator = [2,4]
    for ft in fator:
        amostra = amostragem(copy.deepcopy(img),ft,'{name}-amostragem-{K}{ext}'.format(name=name,K=ft,ext=extension))
    '''
    
    #aqui eu faço quantização de cores e salvo as imagens
    '''
    bits = [2,1]
    for bit in bits:
        nroCores  = (math.pow(2,bit));
        quantizado = quantizacao_uniforme(copy.deepcopy(img),bit,'{name}-quantizacao-{K} cores{ext}'.format(name=name,K=nroCores,ext=extension)) 
    '''
    
    #aqui eu binarizo imagens , nome opcional: '{name}-binarizacao{ext}'.format(name=name,ext=extension)
    
    #img = binarizar(img,None)
    #img2 =  binarizar(img2,None)
    
    
    
    #aqui eu faço as operações#
    
    
    #operacoes_logicas_not(copy.deepcopy(img),'{name}-NOT{ext}'.format(name=name,ext=extension))
    #operacoes_logicas_and(copy.deepcopy(img),copy.deepcopy(img2),'{name}-AND-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
    #operacoes_logicas_or(copy.deepcopy(img),copy.deepcopy(img2),'{name}-OR-{name2}{ext}'.format(name=name,name2=name2,ext=extension))    
    #operacoes_logicas_xor(copy.deepcopy(img),copy.deepcopy(img2),'{name}-XOR-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
    #operacoes_aritmeticas_soma(copy.deepcopy(img),copy.deepcopy(img2),'{name}-SOMA-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
    #operacoes_aritmeticas_subtracao(copy.deepcopy(img),copy.deepcopy(img2),'{name}-SUB-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
    #operacoes_aritmeticas_multiplicacao(copy.deepcopy(img),copy.deepcopy(img2),'{name}-MULT-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
    #operacoes_aritmeticas_divisao(copy.deepcopy(img),copy.deepcopy(img2),'{name}-DIV-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
    #mistura(copy.deepcopy(img),copy.deepcopy(img2),0.8,0.2,0,'{name}-MIST-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
    #distancia_euclidiana([0,0],[1,1])
    #transladar(copy.deepcopy(img),250,250,'{name}-TRANSLADAR{ext}'.format(name=name,ext=extension))
    #rotacionar(copy.deepcopy(img),0,0,3.14,'{name}-ROTACIONAR{ext}'.format(name=name,ext=extension))
    #escalar(copy.deepcopy(img),-1,1.5,'{name}-ESCALAR{ext}'.format(name=name,ext=extension))
    #canais de cores
    #separar_canais(copy.deepcopy(img),'{name}-Blue{ext}'.format(name=name,ext=extension),'{name}-Green{ext}'.format(name=name,ext=extension),'{name}-Red{ext}'.format(name=name,ext=extension))
    #bgr to cmy
    #bgrToCmy(copy.deepcopy(img),'{name}-CMY{ext}'.format(name=name,ext=extension))
    #soma_colorida(copy.deepcopy(img),copy.deepcopy(img2),'{name}-SOMA-COLORIDA-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
    #basicDithering(copy.deepcopy(img),'{name}-basicDith{ext}'.format(name=name,ext=extension))
    #randomDithering(copy.deepcopy(img),'{name}-randomDith{ext}'.format(name=name,ext=extension))
    #AlgoritmoOrdenadoPeriodicoDispersoDithering(copy.deepcopy(img),'{name}-AOPDDith{ext}'.format(name=name,ext=extension))
    #AlgoritmoOrdenadoPeriodicoAglomeradoDithering(copy.deepcopy(img),'{name}-AOPADith{ext}'.format(name=name,ext=extension))
    #AlgoritmoAperiodicoDithering(copy.deepcopy(img),'{name}-AAPADith{ext}'.format(name=name,ext=extension))
    #filtro_media(copy.deepcopy(img),7,'{name}-FiltroMedia{ext}'.format(name=name,ext=extension))
    #filtro_gaussiano(copy.deepcopy(img),'{name}-FiltroGaussiano{ext}'.format(name=name,ext=extension))
    #filtro_mediana(copy.deepcopy(img),'{name}-FiltroMediana{ext}'.format(name=name,ext=extension))
    #realce_negacao(copy.deepcopy(img),'{name}-Negacao{ext}'.format(name=name,ext=extension))
    #realce_contraste(copy.deepcopy(img),100,255,'{name}-Contraste{ext}'.format(name=name,ext=extension))
    #realce_gama(copy.deepcopy(img),1,0.2,'{name}-Gama{ext}'.format(name=name,ext=extension))
    #realce_linear(copy.deepcopy(img),2,0,'{name}-Linear{ext}'.format(name=name,ext=extension))
    #realce_logaritmico(copy.deepcopy(img),'{name}-Logaritmico{ext}'.format(name=name,ext=extension))
    #realce_quadratico(copy.deepcopy(img),'{name}-Quadratico{ext}'.format(name=name,ext=extension))
    #realce_raiz(copy.deepcopy(img),'{name}-Raiz{ext}'.format(name=name,ext=extension))
    #deteccao_pontos(copy.deepcopy(img),250,'{name}-DeteccaoPontos{ext}'.format(name=name,ext=extension))
    
    #para mostrar os resultados das operacoes usando janelas do opencv (nao mto confiavel, as vezes fica errado na janela mas certo em arquivo)
    
    #cv2.imshow("original",img)
    #cv2.imshow("new",img1)
    #print("fazendo a diferença...")
    #img2 = operacoes_aritmeticas_subtracao(copy.deepcopy(img),copy.deepcopy(imgR1),None)
    #cv2.imshow("difference",img2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    
    #atividade de histogramas
    x = [] 
    for i in range(0,256):
        x.append(i)
        
    hist = histograma(copy.deepcopy(img))
    #hist = histograma_cumulativo(copy.deepcopy(img))
    #hist = histograma_equalizado(copy.deepcopy(img),'{name}-Equalizado{ext}'.format(name=name,ext=extension))
    histograma_alongado(copy.deepcopy(img),50,250,'{name}-Alongado{ext}'.format(name=name,ext=extension))
    #histograma_especificado(copy.deepcopy(img),copy.deepcopy(img2),'{name}-Especificado{ext}'.format(name=name,ext=extension))
    plt.bar(x,hist,1)
    plt.ylabel('Pixels');
    
    
    