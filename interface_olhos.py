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

#carregar as ids
id = np.load('id.npy')

TEST_DIR = 'data/test'
IMG_SIZE_OLHOS = 60
LR = 5e-4
nInd =  len(id) #numero de individuos do banco de imagens

NOME_MODELO_OLHOS = 'FotoFaceO-{}-{}.model'.format(LR, '3Conv_2fully')

width, height = 320, 240
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
def valida():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    ts = datetime.datetime.now()
    nome = TEST_DIR + "\\"+ "valida.{}.jpg".format(ts.strftime("%d%m%Y%H%M%S"))
    nome_olhos = TEST_DIR + "\\"+ "olhos.{}.jpg".format(ts.strftime("%d%m%Y%H%M%S"))
    cv2.imwrite(nome_olhos, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imwrite(nome, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cortar_olhos(nome_olhos)
    cortar(nome)

def cortar(nome):
    img = cv2.imread(nome,cv2.IMREAD_COLOR)
    img = img[20:230,55:265]
    cv2.imwrite(nome, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    showImg(nome, myCanvas)
    
def cortar_olhos(nome):
    img = cv2.imread(nome,cv2.IMREAD_COLOR)
    img = img[80:180,110:210]
    cv2.imwrite(nome, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    validar_olhos(nome)

def identifica(model_out):
    nomeid = str(id[np.argmax(model_out)][2])
    snomeid = str(id[np.argmax(model_out)][3])
    percid = np.max(model_out)*100
    if (percid > 80):
        acesso = 'ACESSO GARANTIDO'
    else:
        acesso = 'ACESSO NEGADO'
        nomeid = 'NÃO'
        snomeid = 'LOCALIZADO'
    return NOME.set(nomeid), SobreNome.set(snomeid), Perc.set(acesso)
    
def label_img(img):
    word_label = img.split('.')[-3]
    label = np.zeros(nInd)
    for linha in range(len(id)):
        if (id[linha][1] == word_label):
            label[[linha][0]] = 1
    return label

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
    
def validar_olhos(nome):
    
    test_data = criar_test_data_olhos(nome)
    
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
    model = tflearn.DNN(convnet, tensorboard_dir='log')
    
    
    if os.path.exists('{}.meta'.format(NOME_MODELO_OLHOS)):
        model.load(NOME_MODELO_OLHOS)
        
    test_data = np.load('test_data_olhos.npy')    
        
    img_data = test_data[0]
    data = img_data.reshape(IMG_SIZE_OLHOS,IMG_SIZE_OLHOS,1)
    model_out = model.predict([data])[0]
    print(str(model_out))
    identifica(model_out)

      
janela = tk.Tk()
janela.title("Aplicação")
janela.geometry("660x450+1+1")

NOME = tk.StringVar()
SobreNome = tk.StringVar()
Perc = tk.StringVar()
resultado = tk.DoubleVar()
resultado_olhos = tk.DoubleVar()

tValida = tk.Label(janela, text="VALIDA 1.4", fg="red", font=("calibri", 18))
tValida.grid(row=0, column=2, columnspan=2, pady=20)

botaoValidar = tk.Button(janela, text = "Validar", fg = "black", width=15, command = valida)
botaoValidar.grid(row=6, column=6, columnspan=2, pady=20)

lmain = tk.Label(janela)
lmain.grid(row=1, column=1, columnspan=3, rowspan=3, padx=20)

myCanvas = tk.Canvas(janela, width=210, height=210)
myCanvas.grid(row=1, column=4, columnspan=3, rowspan=3, padx=20)

lnome = tk.Label(janela, textvariable=NOME, fg="gray", text="calibri", font=("calibri", 15))
lnome.grid(row=4, column=1, columnspan=3)

lsobrenome = tk.Label(janela, textvariable=SobreNome, fg="gray", text="calibri", font=("calibri", 15))
lsobrenome.grid(row=5, column=1,columnspan=3)

lperc = tk.Label(janela, textvariable=Perc, fg="red", text="calibri", font=("calibri", 15))
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