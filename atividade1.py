import numpy.matlib
import numpy as np
import math 
import copy
import cv2

##############Amostragem#######################
def amostragem(img, n):
    amostra = [lin[::n] for lin in img[::n]]
    return np.array(amostra)
#################Quantização############################
def quantizacao(img,k):
    quantized =img.copy()

    rows =img.shape[0]
    cols = img.shape[1]
    for i in range(rows):
        for j in range(cols):
            quantized[i,j] =(math.pow(2,k)-1)*np.float32((img[i,j] - img.min())/(img.max()-img.min()))
            quantized[i,j] = np.round(quantized[i,j])*int(256/math.pow(2,k))
    return quantized
###################Operacões Aritméticas Lógicas AND###########################
def img1AndImg2(img1,img2):
    rows,cols = img1.shape
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    img = np.zeros((rows,cols),np.float32)
    for i in range(rows):
        for j in range(cols):
            if img1[i,j]*img2[i,j] != 0:
                img[i,j] = 255
    return img
#################Operacões Aritméticas Lógicas OR###############
def img1OrImg2(img1,img2):
    rows,cols = img1.shape
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    img = np.zeros((rows,cols),np.float32)
    for i in range(rows):
        for j in range(cols):
            if img1[i,j]+img2[i,j] != 0:
                img[i,j] = 255
    return img
#############Operacões Aritméticas Lógicas XOR#####################
def img1XorImg2(img1,img2):
    rows = img1.shape[0]
    cols = img1.shape[1]
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    img = np.zeros((rows*cols),dtype = np.float32).reshape((rows,cols))
    for i in range(rows):
        for j in range(cols):
            if img1[i,j]+img2[i,j] == 255:
                img[i,j] = img1[i,j]
    return img
###########Operacões Aritméticas Lógicas NOT######################
def notImg(img):
    rows = img1.shape[0]
    cols = img1.shape[1]
    nImg = np.zeros((rows*cols),dtype = np.float32).reshape((rows,cols))
    for i in range(rows):
        for j in range(cols):
            if img[i,j] == 0:
                nImg[i,j] = 255
            else:
                nImg[i,j] = 0
    return nImg
############Operacões Aritméticas ADIÇÃO#####################
def img1AddImg2(img1, img2):
    rows = img1.shape[0]
    cols = img1.shape[1]
    img = np.zeros((rows,cols),np.float32).reshape((rows,cols))
    for i in range(rows):
        for j in range(cols):
            if img1[i,j]+img2[i,j] > 255:
                img[i,j] = 255 
            else:
                #img[i,j] = img1[i,j]+img2[i,j]
                img[i,j] = (img1[i,j]+img2[i,j])/2
    return img
############Operacões Aritméticas SUBTRAÇÃO#####################
def img1SubImg2(img1,img2):
    rows = img1.shape[0]
    cols = img1.shape[1]
    img = np.zeros((rows,cols),np.float32)
    for i in range(rows):
        for j in range(cols):
            if img1[i,j] - img2[i,j] < 0:
                img[i,j] = 0
            else:
                img[i,j] = img1[i,j] - img2[i,j]
    return img
##########Operacões Aritméticas MULTIPLICAÇÃO#####################
def imgMultScalar(img1,scalar):
    rows = img1.shape[0]
    cols = img1.shape[1]
    img = np.zeros((rows,cols),np.float32)
    for i in range(rows):
        for j in range(cols):
            if img1[i,j]*scalar > 255:
                img[i,j] = 255
            elif img1[i,j]*scalar < 0:
                img[i,j] = 0
            else:
                img[i,j] = img1[i,j]*scalar
    return img
##########Operacões Aritméticas DIVISAO#################
def img1DivImg2(img1,img2):
    rows = img1.shape[0]
    cols = img1.shape[1]
    img = np.zeros((rows,cols),np.float32)
    for i in range(rows):
        for j in range(cols):
            if(img2[i,j] != 0):
                img[i,j] = img1[i,j]/img2[i,j]
            else:
                img[i,j] = 0
    return img
###########Mistura de Duas Imagens###################
def img1MixImg2(img1,img2,p1,p2):
    rows = img1.shape[0]
    cols = img1.shape[1]
    img = np.zeros((rows,cols),np.float32).reshape((rows,cols))
    for i in range(rows):
        for j in range(cols):
            if img1[i,j]+img2[i,j] > 255:
                img[i,j] = 255 
            else:
                #img[i,j] = img1[i,j]+img2[i,j]
                img[i,j] = (img1[i,j]*p1+img2[i,j]*p2)/(p1+p2)
    return img
############Distancia Euclidiana#######################
def euclideanDistance(p1,p2):
    #((x1-x2)+(y1-y2))^1/2
    return ((p1[0]-p2[0])+(p1[1]-p2[1]))**(1/2)
############Mudar a posição de uma Imagem#####################
def translationMatrix(tx,ty,px,py):
    transformationMatrix = [[1,0,tx],
                            [0,1,ty],
                            [0,0,1]]
    pixelMatrix = [[px],
                   [py],
                   [1]]
    return np.dot(transformationMatrix,pixelMatrix)
def imgTranslation(img1,fx,fy):
    rows = img1.shape[0]
    cols = img1.shape[1]
    img = np.zeros((rows,cols),np.float32).reshape((rows,cols))
    for i in range(rows):
        for j in range(cols):
            newPosition = translationMatrix(fx,fy,i,j)
            if(newPosition[0][0] < rows and newPosition[1][0] < cols):
                img[newPosition[0][0],newPosition[1][0]] = img1[i][j]
    return img
###########Mudar escala de uma imagem###################
def scaleMatrix(sx,sy,px,py):
    transformationMatrix = [[sx,0,0],
                            [0,sy,0],
                            [0,0,1]]
    pixelMatrix = [[px],
                   [py],
                   [1]]
    return np.dot(transformationMatrix,pixelMatrix)

def imgScale(img1,fx,fy):
    rows = img1.shape[0]
    cols = img1.shape[1]
    img = np.zeros((rows,cols),np.float32).reshape((rows,cols))
    for i in range(rows):
        for j in range(cols):
            newPosition = scaleMatrix(fx,fy,i,j)
            if(newPosition[0][0] < rows and newPosition[1][0] < cols):
                img[newPosition[0][0],newPosition[1][0]] = img1[i][j]
    return img
#########Rotacionar uma imagem######################
def rotationMatrix(angle,px,py):
    angle = math.radians(angle)
    sin = math.sin(angle)
    cos = math.cos(angle)
    transformationMatrix = [[cos,-sin,0],
                            [sin,cos,0],
                            [0,0,1]]
    pixelMatrix = [[px],
                   [py],
                   [1]]
    return np.dot(transformationMatrix,pixelMatrix)
def imgRotation(img1,angle):
    rows = img1.shape[0]
    cols = img1.shape[1]
    img = copy.deepcopy(np.float32(img1))
    for i in range(rows):
        for j in range(cols):
            img[i,j] = 0
    img = np.concatenate((img,img),axis=0)
    img = np.concatenate((img,img),axis=1)
    img = np.concatenate((img,img),axis=0)
    img = np.concatenate((img,img),axis=-1)
    
    for i in range(rows):
        for j in range(cols):
            position = rotationMatrix(angle,i,j)
            img[int(round(position[0][0])),int(round(position[1][0]))] = img1[i,j]
    return img
################Amostragem########################
'''
fator = [2,4]
for ft in fator:
    img1 = cv2.imread("cartao2.jpg",0)
    amostra = amostragem(img1,ft)
    
    cv2.imshow("amostra",amostra)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
################Quantização########################
'''
cores = [2,8]
for cor in cores:
    img1 = cv2.imread("cartao2.jpg",0)
    quantizacao = quantizacao(img1,cor)
    
    cv2.imshow("quantização",quantizacao)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
####################AND############################
'''
img1 = cv2.imread("poligono.png",0)
img2 = cv2.imread("triangulo.jpg",0)
newImg = img1AndImg2(img1,img2)

cv2.imshow("AND",newImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
####################OR############################
'''
img1 = cv2.imread("poligono.png",0)
img2 = cv2.imread("triangulo.jpg",0)
newImg = img1OrImg2(img1,img2)

cv2.imshow("OR",newImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
####################XOR############################
'''
img1 = cv2.imread("poligono.png",0)
img2 = cv2.imread("triangulo.jpg",0)
newImg = img1XorImg2(img1,img2)

cv2.imshow("XOR",newImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
####################NOT############################
'''
img1 = cv2.imread("poligono.png",0)
img2 = cv2.imread("triangulo.jpg",0)
newImg1 = notImg(img1)
newImg2 = notImg(img2)
cv2.imshow("NOT1",newImg1)
cv2.imshow("NOT2",newImg2)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

img2 = cv2.imread("planta.jpeg",0)

newImg = imgRotation(copy.deepcopy(img2),-45)
cv2.imshow('Rotation',newImg)
cv2.waitKey(0)
cv2.destroyAllWindows()





















