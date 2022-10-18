from platform import java_ver
from random import randint
import numpy as np
import math
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class TestSubject():
    def __init__(self,file,model: Dict):
        content = [data.strip().split() for data in open(file).readlines()]
        self.model = model
        
        self.samples_by_class = {}
        for key in model.keys():
            self.samples_by_class[key] = []
        [self.samples_by_class[data[-1]].append([float(number) for number in data[0:-1]]) for data in content]
        
        self.samples = np.array([([float(number) for number in data[0:-1]],self.model[data[-1]]) for data in content], dtype=object)
    
    def getAllSamples(self) -> List[Tuple]:
        return self.samples
    
    def getModel(self):
        return self.model
    
    def getClasses(self):
        return self.model.keys()
    
    def getSamplesByClass(self):
        return self.samples_by_class
    
    def drawSamples(self,weight) -> List[Tuple]:
        #np.random.seed(math.floor(time))
        group = []
        test_group = []
        for key in self.samples_by_class.keys():
            samples_selected = np.array(self.samples_by_class[key])
            np.random.shuffle(samples_selected)
            chosen = samples_selected[:math.floor(samples_selected.size*weight)]
            test = samples_selected[-(math.floor(samples_selected.size*weight)-2):]
            [group.append((np.array(sample),np.array(self.model[key]))) for sample in chosen]
            [test_group.append((np.array(sample),np.array(self.model[key]))) for sample in test]

        group = np.array(group, dtype=object)
        np.random.shuffle(group)

        test_group = np.array(test_group, dtype=object)
        np.random.shuffle(test_group)

        return np.array([sample[0] for sample in group]),np.array([sample[1] for sample in group]),np.array([sample[0] for sample in test_group]),np.array([sample[1] for sample in test_group])
        
    
    def translateResponse(self,response):
        return list(self.model.keys())[list(self.model.values()).index(response)]
    
class NeuralNetwork():
    
    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)
        
        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        #applying the sigmoid function
        return np.array([1 / (1 + np.exp(-i)) for i in x])

    def translateSigmoid(self, x):
        return np.array([1 if i==np.argmax(x, axis=0) else 0 for i in range(len(x))])
         
    def step(self, x):
        #computing derivative to the Sigmoid function
        return np.array([1 if i>=0 else 0 for i in x])

    def train(self, max_it, alpha, X, d, function):
        # Pesos Sorteados
        w = np.random.randint(-1,1,(d[0].size,X[0].size))
        b = np.random.randint(-1,1,(d[0].size))
        
        t = 1
        E = 1

        error_epoch = []
        while (t < max_it and E > 0):
            E = 0
            for i in range(0,len(X)):
                y = function(w.dot(X[i])+b)
                e = d[i] - y
            
                w = w + (alpha*np.array([e]).transpose()*np.array([X[i]]))
                b = b + (alpha*e)
                
                E = E + np.sum(np.square(e))
            error_epoch.append(E)
            t = t + 1
        return w,b, error_epoch

    def test(self, w, b, Xt, dt, function):
        errors = []
        for i in range(0,len(Xt)):
            y = function(w.dot(Xt[i])+b)
            comparison = dt[i]==self.translateSigmoid(y)
            e = comparison.all()
            errors.append(e)
        return errors
          


if __name__ == "__main__":

    #initializing the neuron class
    neural_network = NeuralNetwork()

    #training data consisting of 4 examples--3 input values and 1 output
    subject = TestSubject("column_3C.dat",{"DH":[0,0,1],"SL":[0,1,0],"NO":[1,0,0]})

    #training taking place
    X,d,Xt,dt = subject.drawSamples(0.7)
    w,b,err_epoch = neural_network.train(alpha=0.1, max_it=100, X=X, d=d, function=neural_network.sigmoid)
    plt.plot(err_epoch)
    plt.ylabel('Erro do treino')
    plt.show()

    y_found = neural_network.test(w,b,Xt,dt, function=neural_network.sigmoid)
    acc = sum(y_found)/len(y_found)

    print(f"Acurácia: {acc}")
    
    
