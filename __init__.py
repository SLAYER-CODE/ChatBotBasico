from nis import cat
from pyexpat import model
from statistics import mode
from tkinter import W
from urllib.error import ContentTooShortError
from urllib.robotparser import RobotFileParser
from xml.dom.minidom import Document

from nltk.stem.lancaster import LancasterStemmer
from main import main
from utils import utils
import numpy
import tflearn
import tensorflow 
from tensorflow.python.framework import ops

import json
import random
import pickle

obj = utils("contenido.json")
try:
    with open("var.pickle","rb") as archPickle:
        obj.chars,obj.tags,entrenamiento,salida = pickle.load(archPickle)

except FileNotFoundError:
    obj.serialize()
    obj.order()

    entrenamiento = []
    salida = []

    salidaVacia = [0 for _ in range(len(obj.tags))]

    for x , document in enumerate(obj.auxX):
        cubeta = []
        auxPalabra  = [utils._stemmer.stem(w.lower())  for w in document]
        for w in obj.chars:
            if w in auxPalabra:
                cubeta.append(1)
            else:
                cubeta.append(0)
        filaSalida = salidaVacia[:]
        filaSalida[obj.tags.index(obj.auxY[x])] = 1
        entrenamiento.append(cubeta)
        salida.append(filaSalida)

    entrenamiento = numpy.array(entrenamiento)
    salida = numpy.array(salida)
    with open("var.pickle","wb") as ArchPickle:
        pickle.dump((obj.chars,obj.tags,entrenamiento,salida),ArchPickle)
    

try:
    modelo = tflearn.DNN()
    modelo.load("modelo.tfOne")
    assert input("Decea generar nuevamente el modulo? [Y/N]").lower()=="n"
except Exception:
        
    ops.reset_default_graph()
    red = tflearn.input_data(shape=[None,len(entrenamiento[0])])
    red = tflearn.fully_connected(red,40)
    red = tflearn.fully_connected(red,40)

    #Conexion entre todas las salidas y entradas (full)
    red = tflearn.fully_connected(red,len(salida[0]),activation="softmax")

    #Obtener probabilidades para clasificar nuestra respuesta en los tags
    red = tflearn.regression(red)

    #Creacion de semi Modelo
    modelo = tflearn.DNN(red)
    
    #Generando el modulo de itineracion!
    modelo.fit(entrenamiento,salida,n_epoch=10000,batch_size=10,show_metric=True)
    modelo.save("modelo.tfOne")
    
main(obj,modelo)