# -*- coding: utf-8 -*-
"""
Feito por Pedropva em 28/08/2018, meu aniversário :p
"""

import numpy as np
import os,sys,math,copy,random,cv2
import matplotlib.pyplot as plt
sys.path.append('./codigos/')#importa os outros codigos de cada atividade
import amostragemEQuantizacao,operacoesLogicas,operacoesAritmeticas,transformacoes
import conversoesCores,dithering,filtros,realces,histogramas,deteccaoDescontinuidades


filename = 'pei.jpeg'
img = cv2.imread(filename,0)
name, extension = os.path.splitext(filename)

filename2 = 'reee.jpg'
img2 = cv2.imread(filename2,0)
name2, extension2 = os.path.splitext(filename2)


#Atividade de amostragem e Quantizacao
"""
fator = [2,4]
for ft in fator:
    amostra = amostragemEQuantizacao.amostragem(copy.deepcopy(img),ft,'./resultados/{name}-amostragem-{K}{ext}'.format(name=name,K=ft,ext=extension))
bits = [2,1]
for bit in bits:
    nroCores  = (math.pow(2,bit));
    quantizado = amostragemEQuantizacao.quantizacao_uniforme(copy.deepcopy(img),bit,'./resultados/{name}-quantizacao-{K} cores{ext}'.format(name=name,K=nroCores,ext=extension)) 
"""

#aqui eu binarizo imagens pra fazer as opescoes logicas
#nome opcional caso queira salvar as imgs binarizadas: './resultados/{name}-binarizacao{ext}'.format(name=name,ext=extension)
#img = amostragemEQuantizacao.binarizar(img,None)
#img2 =  amostragemEQuantizacao.binarizar(img2,None)





#Atividade de operacoes logicas
#operacoesLogicas.Not(copy.deepcopy(img),'./resultados/{name}-NOT{ext}'.format(name=name,ext=extension))
#operacoesLogicas.And(copy.deepcopy(img),copy.deepcopy(img2),'./resultados/{name}-AND-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
#operacoesLogicas.Or(copy.deepcopy(img),copy.deepcopy(img2),'./resultados/{name}-OR-{name2}{ext}'.format(name=name,name2=name2,ext=extension))    
#operacoesLogicas.Xor(copy.deepcopy(img),copy.deepcopy(img2),'./resultados/{name}-XOR-{name2}{ext}'.format(name=name,name2=name2,ext=extension))

#Atividade de operacoes aritmeticas
#operacoesAritmeticas.soma(copy.deepcopy(img),copy.deepcopy(img2),'./resultados/{name}-SOMA-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
#operacoesAritmeticas.subtracao(copy.deepcopy(img),copy.deepcopy(img2),'./resultados/{name}-SUB-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
#operacoesAritmeticas.multiplicacao(copy.deepcopy(img),copy.deepcopy(img2),'./resultados/{name}-MULT-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
#operacoesAritmeticas.divisao(copy.deepcopy(img),copy.deepcopy(img2),'./resultados/{name}-DIV-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
#operacoesAritmeticas.mistura(copy.deepcopy(img),copy.deepcopy(img2),0.8,0.2,0,'./resultados/{name}-MIST-{name2}{ext}'.format(name=name,name2=name2,ext=extension))
#operacoesAritmeticas.distancia_euclidiana([0,0],[1,1])
#operacoesAritmeticas.soma_colorida(copy.deepcopy(img),copy.deepcopy(img2),'./resultados/{name}-SOMA-COLORIDA-{name2}{ext}'.format(name=name,name2=name2,ext=extension))

#Atividade de transformacoes pixel a pixel
#transformacoes.transladar(copy.deepcopy(img),250,250,'./resultados/{name}-TRANSLADAR{ext}'.format(name=name,ext=extension))
#transformacoes.rotacionar(copy.deepcopy(img),0,0,3.14,'./resultados/{name}-ROTACIONAR{ext}'.format(name=name,ext=extension))
#transformacoes.escalar(copy.deepcopy(img),-1,1.5,'./resultados/{name}-ESCALAR{ext}'.format(name=name,ext=extension))

#Atividade de sistemas de cores
#conversoesCores.separar_canais(copy.deepcopy(img),'./resultados/{name}-Blue{ext}'.format(name=name,ext=extension),'./resultados/{name}-Green{ext}'.format(name=name,ext=extension),'./resultados/{name}-Red{ext}'.format(name=name,ext=extension))
#conversoesCores.bgrToCmy(copy.deepcopy(img),'./resultados/{name}-CMY{ext}'.format(name=name,ext=extension))
#conversoesCores.bgrToYuv(copy.deepcopy(img),'./resultados/{name}-YUV{ext}'.format(name=name,ext=extension))
#conversoesCores.bgrToYCrCb(copy.deepcopy(img),1,'./resultados/{name}-YCRCB{ext}'.format(name=name,ext=extension))
#conversoesCores.bgrToYiQ(copy.deepcopy(img),'./resultados/{name}-YIQ{ext}'.format(name=name,ext=extension))
#conversoesCores.bgrTorgb(copy.deepcopy(img),'./resultados/{name}-RGB{ext}'.format(name=name,ext=extension))

#Atividade de dithering
#dithering.basicDithering(copy.deepcopy(img),'./resultados/{name}-basicDith{ext}'.format(name=name,ext=extension))
#dithering.randomDithering(copy.deepcopy(img),'./resultados/{name}-randomDith{ext}'.format(name=name,ext=extension))
#dithering.AlgoritmoOrdenadoPeriodicoDisperso(copy.deepcopy(img),'./resultados/{name}-AOPDDith{ext}'.format(name=name,ext=extension))
#dithering.AlgoritmoOrdenadoPeriodicoAglomerado(copy.deepcopy(img),'./resultados/{name}-AOPADith{ext}'.format(name=name,ext=extension))
#dithering.AlgoritmoAperiodico(copy.deepcopy(img),'./resultados/{name}-AAPADith{ext}'.format(name=name,ext=extension))

#Atividade de filtros
#filtros.media(copy.deepcopy(img),7,'./resultados/{name}-FiltroMedia{ext}'.format(name=name,ext=extension))
#filtros.gaussiano(copy.deepcopy(img),'./resultados/{name}-FiltroGaussiano{ext}'.format(name=name,ext=extension))
#filtros.mediana(copy.deepcopy(img),'./resultados/{name}-FiltroMediana{ext}'.format(name=name,ext=extension))

#Atividade de realces
#realces.negacao(copy.deepcopy(img),'./resultados/{name}-Negacao{ext}'.format(name=name,ext=extension))
#realces.contraste(copy.deepcopy(img),100,255,'./resultados/{name}-Contraste{ext}'.format(name=name,ext=extension))
#realces.gama(copy.deepcopy(img),1,0.2,'./resultados/{name}-Gama{ext}'.format(name=name,ext=extension))
#realces.linear(copy.deepcopy(img),2,0,'./resultados/{name}-Linear{ext}'.format(name=name,ext=extension))
#realces.logaritmico(copy.deepcopy(img),'./resultados/{name}-Logaritmico{ext}'.format(name=name,ext=extension))
#realces.quadratico(copy.deepcopy(img),'./resultados/{name}-Quadratico{ext}'.format(name=name,ext=extension))
#realces.raiz(copy.deepcopy(img),'./resultados/{name}-Raiz{ext}'.format(name=name,ext=extension))

#Atividade de histogramas
"""
x = [] 
for i in range(0,256):
    x.append(i)
    
#hist = histogramas.histograma(copy.deepcopy(img))
#hist = histogramas.cumulativo(copy.deepcopy(img))
hist = histogramas.equalizado(copy.deepcopy(img),'./resultados/{name}-Equalizado{ext}'.format(name=name,ext=extension))
histogramas.alongado(copy.deepcopy(img),50,250,'./resultados/{name}-Alongado{ext}'.format(name=name,ext=extension))
histogramas.especificado(copy.deepcopy(img),copy.deepcopy(img2),'./resultados/{name}-Especificado{ext}'.format(name=name,ext=extension))
plt.bar(x,hist,1)
plt.ylabel('Pixels');

plt.show()
"""

#Atividade de deteccao de descontinuidades
deteccaoDescontinuidades.pontos(copy.deepcopy(img),200,'./resultados/{name}-DeteccaoPontos{ext}'.format(name=name,ext=extension))
#angulo = 0
#deteccaoDescontinuidades.retas(copy.deepcopy(img),angulo,200,'./resultados/{name}-DeteccaoRetas{angulo}{ext}'.format(name=name,angulo=angulo,ext=extension))
#angulo = 45
#deteccaoDescontinuidades.retas(copy.deepcopy(img),angulo,200,'./resultados/{name}-DeteccaoRetas{angulo}{ext}'.format(name=name,angulo=angulo,ext=extension))
#angulo = 90
#deteccaoDescontinuidades.retas(copy.deepcopy(img),angulo,200,'./resultados/{name}-DeteccaoRetas{angulo}{ext}'.format(name=name,angulo=angulo,ext=extension))
#angulo = -45
#deteccaoDescontinuidades.retas(copy.deepcopy(img),angulo,200,'./resultados/{name}-DeteccaoRetas{angulo}{ext}'.format(name=name,angulo=angulo,ext=extension))

deteccaoDescontinuidades.roberts(copy.deepcopy(img),200,'./resultados/{name}-DeteccaoRoberts{ext}'.format(name=name,ext=extension))
deteccaoDescontinuidades.prewitt(copy.deepcopy(img),200,'./resultados/{name}-DeteccaoPrewitt{ext}'.format(name=name,ext=extension))
deteccaoDescontinuidades.sobel(copy.deepcopy(img),200,'./resultados/{name}-DeteccaoSobel{ext}'.format(name=name,ext=extension))
deteccaoDescontinuidades.laplaciano(copy.deepcopy(img),200,'./resultados/{name}-DeteccaoLaplaciano{ext}'.format(name=name,ext=extension))


#para mostrar os resultados das operacoes usando janelas do opencv (nao mto confiavel, as vezes fica errado na janela mas certo em arquivo)
#cv2.imshow("original",img)
#cv2.imshow("new",img1)
#print("fazendo a diferença...")
#img2 = operacoes_aritmeticas_subtracao(copy.deepcopy(img),copy.deepcopy(imgR1),None)
#cv2.imshow("difference",img2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



