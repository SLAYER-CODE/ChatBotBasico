import json
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer

class utils(object):
    _stemmer = LancasterStemmer()
    def __init__(self,content:str):
        
        with open("contenido.json",encoding="utf-8") as arch:
            self.dat = json.load(arch)
    
            
        self.chars=[]
        self.tags=[]
        self.auxX=[]
        self.auxY=[]
    def serialize(self):
        for content in self.dat["contenido"]:
            for patrones in content["patrones"]:
                auxPalabra = nltk.word_tokenize(patrones)
                self.chars.extend(auxPalabra)
                self.auxX.append(auxPalabra)
                self.auxY.append(content["tag"])
                if content["tag"] not in self.tags:
                    self.tags.append(content["tag"])
    def order(self):
        self.chars = [utils._stemmer.stem(w.lower())  for w in self.chars if  w != "?"]
        self.chars = sorted(list(set(self.chars)))
        self.tags = sorted(self.tags)
