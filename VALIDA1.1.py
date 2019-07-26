# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 19:02:51 2017

@author: ee4yo
"""

import tkinter as tk
import datetime
import cv2
from PIL import Image, ImageTk                
import numpy as np         
import os                  # Lida com diretorios
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import time

#carregar as ids
id = np.load('id.npy')

TEST_DIR = 'data/test'
AVA_DIR = 'avatar/'
IMG_SIZE = 100
IMG_SIZE_OLHOS = 60
LR = 5e-4
nInd =  len(id) #numero de individuos do banco de imagens

NOME_MODELO = 'FotoFace-{}-{}.model'.format(LR, '3Conv_2fully')
NOME_MODELO_OLHOS = 'FotoFaceO-{}-{}.model'.format(LR, '3Conv_2fully')

width, height = 320, 240
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def resetar():
    showImg("face2.png", myCanvas)
    return NOME.set(""), SobreNome.set(""), Perc.set("")
    
def valida():
    inicio = time.time()
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    ts = datetime.datetime.now()
    nome = TEST_DIR + "\\"+ "valida.{}.jpg".format(ts.strftime("%d%m%Y%H%M%S"))
    nome_olhos = TEST_DIR + "\\"+ "olhos.{}.jpg".format(ts.strftime("%d%m%Y%H%M%S"))
    cv2.imwrite(nome, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imwrite(nome_olhos, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cortar(nome,nome_olhos)
    fim = time.time()
    print("Tempo de execução: " + str(fim - inicio))

def cortar(nome,nome_olhos):
    img = cv2.imread(nome,cv2.IMREAD_COLOR)
    img_olhos = cv2.imread(nome_olhos,cv2.IMREAD_COLOR)
    img = img[20:230,55:265]
    img_olhos = img_olhos[80:180,110:210]
    cv2.imwrite(nome, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imwrite(nome_olhos, img_olhos, [cv2.IMWRITE_JPEG_QUALITY, 100])
    showImg(nome, myCanvas)
    model_out=rede1(nome,model)
    model_out_olhos=rede2(nome_olhos,model_olhos)
    identifica(model_out, model_out_olhos)

def identifica(model_out, model_out_olhos):
    classe = id[np.argmax(model_out)][0]
    classe_olhos = id[np.argmax(model_out_olhos)][0]
    nomeid = str(id[np.argmax(model_out)][2])
    snomeid = str(id[np.argmax(model_out)][3])
    perc = np.max(model_out)*100
    perc_olhos = np.max(model_out_olhos)*100
    
    if (classe == classe_olhos and perc > 90 and perc_olhos > 60):
        acesso = 'ACESSO GARANTIDO'
    else:
        acesso = 'Tente outra vez'
        nomeid = 'NÃO LOCALIZADO'
        snomeid = 'ACESSO NEGADO'
    return NOME.set(nomeid), SobreNome.set(snomeid), Perc.set(acesso)
     
def label_img(img):
    word_label = img.split('.')[-3]
    label = np.zeros(nInd)
    for linha in range(len(id)):
        if (id[linha][1] == word_label):
            label[[linha][0]] = 1
    return label

def criar_test_data(nome):
    testing_data = []
    img = cv2.imread(nome,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    testing_data.append([np.array(img)])
        
    np.save('test_data.npy', testing_data)
    return testing_data

def criar_test_data_olhos(nome):
    testing_data = []
    img = cv2.imread(nome,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE_OLHOS,IMG_SIZE_OLHOS))
    testing_data.append([np.array(img)])
        
    np.save('test_data_olhos.npy', testing_data)
    return testing_data

def showImg(n, canvas):
    render = ImageTk.PhotoImage(file=n)
    myCanvas.image = render
    myCanvas.create_image(0,0, image = render, anchor = "nw")
    
def rede1(nome,model):
    
    test_data = criar_test_data(nome)
        
    test_data = np.load('test_data.npy')    
        
    img_data = test_data[0]
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]
    print(model_out)
    return model_out

    
def rede2(nome_olhos, model):
        
    test_data_olhos = criar_test_data_olhos(nome_olhos)  
        
    test_data_olhos = np.load('test_data_olhos.npy')    
        
    img_data_olhos = test_data_olhos[0]
    data_olhos = img_data_olhos.reshape(IMG_SIZE_OLHOS,IMG_SIZE_OLHOS,1)
    model_out_olhos = model.predict([data_olhos])[0]
    print(model_out_olhos)
    return model_out_olhos


corFundo = '#%02x%02x%02x' % (109, 189, 196)
corBotao = '#%02x%02x%02x' % (236, 121, 100)
corTexto1 = '#%02x%02x%02x' % (50, 50, 50)
          
janela = tk.Tk()
janela.configure(bg=corFundo)
janela.title("VALIDA 1.1")
janela.geometry("640x450+1+1")
    
NOME = tk.StringVar()
SobreNome = tk.StringVar()
Perc = tk.StringVar()
resultado = tk.DoubleVar()
resultado_olhos = tk.DoubleVar()

tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 20, 20, activation='prelu')
convnet = max_pool_2d(convnet, 10)
    
convnet = conv_2d(convnet, 25, 10, activation='prelu')
convnet = max_pool_2d(convnet, 5)
    
convnet = conv_2d(convnet, 30, 5, activation='prelu')
convnet = max_pool_2d(convnet, 5)
    
convnet = fully_connected(convnet, nInd*10, activation='prelu')
convnet = dropout(convnet, 0.8)
    
convnet = fully_connected(convnet, nInd, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, 
                     loss='categorical_crossentropy',
                     batch_size=16,
                     name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log')
    
if os.path.exists('{}.meta'.format(NOME_MODELO)):
    model.load(NOME_MODELO)
    
tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE_OLHOS, IMG_SIZE_OLHOS, 1], name='input')
    
convnet = conv_2d(convnet, 20, 20, activation='prelu')
convnet = max_pool_2d(convnet, 10)
    
convnet = conv_2d(convnet, 25, 10, activation='prelu')
convnet = max_pool_2d(convnet, 5)
    
convnet = conv_2d(convnet, 30, 5, activation='prelu')
convnet = max_pool_2d(convnet, 5)
    
convnet = fully_connected(convnet, nInd*10, activation='prelu')
convnet = dropout(convnet, 0.8)
    
convnet = fully_connected(convnet, nInd, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, 
                     loss='categorical_crossentropy',
                     batch_size=16,
                     name='targets')
model_olhos = tflearn.DNN(convnet, tensorboard_dir='log')
     
if os.path.exists('{}.meta'.format(NOME_MODELO_OLHOS)):
    model_olhos.load(NOME_MODELO_OLHOS)
    
tValida = tk.Label(janela, text="", 
                   bg=corFundo, fg="white", font=("calibri", 12))
tValida.grid(row=0, column=1, columnspan=3, pady=15)

botaoValidar = tk.Button(janela, text = "Validar", 
                         bg=corBotao, fg = corTexto1, width=12, 
                         command = valida,
                         font=("calibri", 11, "bold"))
botaoValidar.grid(row=6, column=4, columnspan=2, pady=20)

botaoReset = tk.Button(janela, text = "Reset", 
                       bg=corBotao, fg = corTexto1, width=12, 
                       command = resetar,
                       font=("calibri", 11, "bold"))
botaoReset.grid(row=6, column=6, columnspan=2, pady=20)

lmain = tk.Label(janela, borderwidth=0.5)
lmain.grid(row=1, column=1, columnspan=3, rowspan=3, padx=20, pady=5)

myCanvas = tk.Canvas(janela, bg=corFundo,
                     bd=0,
                     highlightbackground=corFundo,
                     highlightcolor=corBotao,
                     width=210, height=210)
myCanvas.grid(row=1, column=4, columnspan=3, rowspan=3, padx=20)
showImg("face2.png", myCanvas)

lnome = tk.Label(janela, textvariable=NOME, bg=corFundo, 
                 fg="red", text="calibri", 
                 font=("calibri", 15, "bold"))
lnome.grid(row=4, column=1, columnspan=3)

lsobrenome = tk.Label(janela, textvariable=SobreNome, bg=corFundo, 
                      fg="red", text="calibri",
                      font=("calibri", 15, "bold"))
lsobrenome.grid(row=5, column=1,columnspan=3)

lperc = tk.Label(janela, textvariable=Perc, bg=corFundo, fg="green", 
                 text="calibri", 
                 font=("calibri", 16, "bold"))
lperc.grid(row=6, column=1,columnspan=3)   

def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2.rectangle(cv2image, (90,20), (230,220), (0,0,255))
    cv2.rectangle(cv2image, (110,80), (210,180), (0,0,255))
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, show_frame)
    
show_frame()
janela.mainloop()