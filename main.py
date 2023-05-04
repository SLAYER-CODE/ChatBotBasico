from pyexpat import model
from random import random
from statistics import mode
from typing import List
from unittest import result
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
from utils import utils
import random


def anality(entrada):
    if("#Autenticar"==entrada):
        pass
    elif("#Exit"==entrada):
        pass
    elif("#NotAction"==entrada):
        pass
def main(obj:utils,model:tflearn.DNN):
    while True:
        entrada = input("TU: ")
        anality(entrada)
        cubeta  = [0 for _ in range (len(obj.chars))]
        ProcesIntro = nltk.word_tokenize(entrada)
        ProcesIntro = [utils._stemmer.stem(char.lower()) for char in ProcesIntro]
        for IndvChar in ProcesIntro:
            for i,char in enumerate(obj.chars):
                if char == IndvChar:
                    cubeta[i] = 1
        res  = model.predict([numpy.array(cubeta)])
        resIndex = numpy.argmax(res)
        tag = obj.tags[resIndex]
        for tagAux in obj.dat["contenido"]:
            if tagAux["tag"] == tag:
                response = tagAux["respuestas"]
        print("Bot ", random.choice(response))
