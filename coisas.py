# -*- coding: utf-8 -*-
"""
Feito por Pedropva em 28/08/2018, meu aniversário :p
"""

import numpy as np
import cv2
import os,math, copy
import utils


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



















if __name__ == "__main__":
    
    filename = 'planta.jpeg'
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
    #cv2.imshow('Rotation',newImg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    