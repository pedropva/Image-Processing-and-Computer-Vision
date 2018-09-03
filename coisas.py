# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

# -*- coding: utf-8 -*-
"""
Feito por Pedropva em 28/08/2018, meu aniversário :p
"""

import numpy as np
import cv2
import os,math, copy


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
    
    if img1.shape[0] > img2 .shape[0]:
        rows = img1.shape[0]
        cols = img1.shape[1]
    else:
        rows = img2.shape[0]
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
    
    if img1.shape[0] > img2 .shape[0]:
        rows = img1.shape[0]
        cols = img1.shape[1]
    else:
        rows = img2.shape[0]
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
    
    if img1.shape[0] > img2 .shape[0]:
        rows = img1.shape[0]
        cols = img1.shape[1]
    else:
        rows = img2.shape[0]
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


def operacoes_aritmeticas_soma(img1,img2,filename):
    img = np.float32(img1)
    
    if img1.shape[0] > img2 .shape[0]:
        rows = img1.shape[0]
        cols = img1.shape[1]
    else:
        rows = img2.shape[0]
        cols = img2.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            img[i,j] = img[i,j] + img2[i,j];
            if img[i,j] > 255: img[i,j] = 255;
                
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
    
    cv2.imwrite("reeeTotal.jpg",imgTotal) 
        
        
    return img





















if __name__ == "__main__":
    
    filename = 'reee.jpg'
    img = cv2.imread(filename,1)
    name, extension = os.path.splitext(filename)
    
    filename2 = 'reee-1-NOT.jpg'
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
    
    #img = binarizar(img,'pei.jpg')
    #img2 =  binarizar(img2,None)
    
    
    
    #aqui eu faço as operações lógicas#
    
    #NOT
    #operacoes_logicas_not(copy.deepcopy(img),'{name}-NOT{ext}'.format(name=name,ext=extension))
    #AND
    #operacoes_logicas_and(copy.deepcopy(img),copy.deepcopy(img2),'{name}-AND-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
    #OR
    #operacoes_logicas_or(copy.deepcopy(img),copy.deepcopy(img2),'{name}-OR-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
    #XOR
    #operacoes_logicas_xor(copy.deepcopy(img),copy.deepcopy(img2),'{name}-XOR-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
    #SOMA
    #operacoes_aritmeticas_soma(copy.deepcopy(img),copy.deepcopy(img2),'{name}-SOMA-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
    #canais
    separar_canais(copy.deepcopy(img),'{name}-Blue{ext}'.format(name=name,ext=extension),'{name}-Green{ext}'.format(name=name,ext=extension),'{name}-Red{ext}'.format(name=name,ext=extension))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
