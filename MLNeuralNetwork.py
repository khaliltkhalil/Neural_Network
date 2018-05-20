# Neural Network Implementation from scratch
# Author Khalil Khalil
# 4/30/2018


import numpy as np
import math

# implementation of Multi-Layer neural network
# the activation function is Relu for the hidden unit


def relu(z):
    return np.maximum(z,0)

def reluDeriv(z):
    z[z>=0] = 1
    z[z<0] = 0
    return z

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(y):
    return (np.exp(y))/(np.sum(np.exp(y),axis=0))




class MLNeuralNetwork(object):
    def __init__(self,hidden_layers,alpha_=0.01,lambda_=0,drop_out=False,keep_prob="default",n_iterations=2000,
                 mini_batch_size=256,optimizer="sgd",beta1=0.9,beta2=0.99,epsilon=1e-8):
        self.hidden_layers = hidden_layers
        self.alpha_ = alpha_
        self.lambda_ = lambda_
        self.optimizer = optimizer
        self.updateParameters = {"sgd": self.updateParameters_sgd, "Adam": self.updateParameters_Adam}
        self.beta1 = beta1
        self.beta2 = beta2
        self.Adam_V = {}
        self.Adam_S = {}
        self.Adam_t = 1
        self.epsilon = epsilon
        self.drop_out = drop_out
        if keep_prob == "default":  # the default value is 0.5 for each layer
            self.keep_prob = [0.5]*len(hidden_layers)
        else:
            self.keep_prob = keep_prob
        self.mini_batch_size = mini_batch_size
        self.n_iterations = n_iterations
        self.L = len(hidden_layers)
        self.parameters = {}
        self.cash = {}
        self.paraderiv = {}
        self.costs = []

    def preprocessing(self,X,y):
        X = X.astype(float)
        y = y.astype(int)
        X = X.T
        nk = max(y)+1
        y_hotdot = np.zeros((nk,y.shape[0]))
        for (i,x) in enumerate(y):
            y_hotdot[x,i] = 1
        return (X,y_hotdot)

    def get_mini_batches(self,X,y,seed):

        np.random.seed(seed)
        m = X.shape[1]
        permutation = list(np.random.permutation(m))
        X_shuffled = X[:, permutation]
        y_shuffled = y[:, permutation]
        mini_batches_list = []
        number_complete_mini_batches = int(math.floor(m/self.mini_batch_size))
        for i in range(number_complete_mini_batches):
            X_mini_batch = X_shuffled[:, i*self.mini_batch_size:self.mini_batch_size*(i+1)]
            y_mini_batch = y_shuffled[:, i*self.mini_batch_size:self.mini_batch_size*(i+1)]
            mini_batches_list.append((X_mini_batch, y_mini_batch))

        if m % self.mini_batch_size != 0:
            X_mini_batch = X_shuffled[:, number_complete_mini_batches*self.mini_batch_size:]
            y_mini_batch = y_shuffled[:, number_complete_mini_batches*self.mini_batch_size:]
            mini_batches_list.append((X_mini_batch, y_mini_batch))

        return mini_batches_list


    def forwardProp(self,X):
        Alp = X
        self.cash["A0"] = X
        for l in range(1,self.L+2):
            Wl = self.parameters["W"+str(l)]
            bl = self.parameters["b"+str(l)]
            Zl = np.dot(Wl,Alp) + bl
            if l == self.L+1:
                Al = softmax(Zl)
            else:
                Al = relu(Zl)
                if self.drop_out:
                    DAl = np.random.rand(Al.shape[0],Al.shape[1]) < self.keep_prob[l-1]
                    Al = np.multiply(Al,DAl)
                    Al = Al / self.keep_prob[l-1]

            self.cash["A"+str(l)] = Al
            self.cash["Z"+str(l)] = Zl
            Alp = Al

    def computeCost(self,Alk,Y):
        m = Y.shape[1]
        reg_term = 0

        for l in range(1,self.L+1):
            reg_term = reg_term + (self.lambda_/(2*m)) * np.sum(self.parameters["W"+str(l)]*self.parameters["W"+str(l)])

        #lost_func = (-1 / m) * ( np.sum(Y*np.log(Alk))+ np.sum((1-Y)*np.log(1-Alk))) + reg_term
        lost_func = (-1/m) * np.sum(Y*np.log(Alk)) + reg_term
        return lost_func

    def backProp(self,X,Y):
        m = X.shape[1]
        Al = self.cash["A"+str(self.L+1)]
        dZl = Al - Y
        for l in range(self.L+1,0,-1):
            Alp = self.cash["A"+str(l-1)]  # A1p: A of the previous layer
            Wl = self.parameters["W" + str(l)]
            dWl = (1 / m) * np.dot(dZl,Alp.T) + (self.lambda_/m)*Wl
            dbl = (1 / m) * np.sum(dZl, axis=1, keepdims=True)
            self.paraderiv["dW" + str(l)] = dWl
            self.paraderiv["db" + str(l)] = dbl
            if l == 1:
                break
            Zlp = self.cash["Z" + str(l-1)]
            dZl = reluDeriv(Zlp)*np.dot(Wl.T,dZl)

    def initalizeWeights(self,n0,nk):
        L = [n0] + self.hidden_layers + [nk]
        for l in range(1,self.L+2):
            self.parameters["W"+str(l)] = np.random.randn(L[l],L[l-1]) * np.sqrt(2/L[l-1])
            self.parameters["b"+str(l)] = np.zeros((L[l], 1))

    def initalizeAdam(self,n0,nk):
        self.Adam_t = 1
        L = [n0] + self.hidden_layers + [nk]
        for l in range(1,self.L+2):
            self.Adam_V["VW" + str(l)] = np.zeros((L[l], L[l - 1]))
            self.Adam_V["Vb" + str(l)] = np.zeros((L[l], 1))
            self.Adam_S["SW" + str(l)] = np.zeros((L[l], L[l - 1]))
            self.Adam_S["Sb" + str(l)] = np.zeros((L[l], 1))

    def updateParameters_sgd(self):
        for l in range(1,self.L+2):
            self.parameters["W"+str(l)] = self.parameters["W"+str(l)] - self.alpha_ * self.paraderiv["dW"+str(l)]
            self.parameters["b"+str(l)] = self.parameters["b"+str(l)] - self.alpha_ * self.paraderiv["db"+str(l)]

    def updateParameters_Adam(self):
        Adam_V_corrected = {}
        Adam_S_corrected = {}
        for l in range(1,self.L+2):
            self.Adam_V["VW" + str(l)] = self.beta1 * self.Adam_V["VW" + str(l)] + (1-self.beta1) * self.paraderiv["dW"+str(l)]
            self.Adam_V["Vb" + str(l)] = self.beta1 * self.Adam_V["Vb" + str(l)] + (1 - self.beta1) * self.paraderiv["db" + str(l)]
            self.Adam_S["SW" + str(l)] = self.beta2 * self.Adam_S["SW" + str(l)] + (1 - self.beta2) * np.power(self.paraderiv["dW" + str(l)],2)
            self.Adam_S["Sb" + str(l)] = self.beta2 * self.Adam_S["Sb" + str(l)] + (1 - self.beta2) * np.power(self.paraderiv["db" + str(l)],2)
            Adam_V_corrected["VW" + str(l)] = self.Adam_V["VW" + str(l)] / (1 - self.beta1**self.Adam_t)
            Adam_V_corrected["Vb" + str(l)] = self.Adam_V["Vb" + str(l)] / (1 - self.beta1 ** self.Adam_t)
            Adam_S_corrected["SW" + str(l)] = self.Adam_S["SW" + str(l)] / (1 - self.beta2 ** self.Adam_t)
            Adam_S_corrected["Sb" + str(l)] = self.Adam_S["Sb" + str(l)] / (1 - self.beta2 ** self.Adam_t)
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - self.alpha_ * Adam_V_corrected["VW" + str(l)]/(np.sqrt(Adam_S_corrected["SW"+str(l)])+self.epsilon)
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - self.alpha_ * Adam_V_corrected["Vb" + str(l)]/(np.sqrt(Adam_S_corrected["Sb"+str(l)])+self.epsilon)
        self.Adam_t += 1

    def train(self,X,y):
        (X,Y) = self.preprocessing(X,y)
        seed = 0
        n0 = X.shape[0]
        nk = Y.shape[0]
        self.initalizeWeights(n0,nk)
        if self.optimizer == "Adam":
            self.initalizeAdam(n0,nk)
        for i in range(self.n_iterations):
            seed += 1
            mini_batches_list = self.get_mini_batches(X, Y, seed)
            for (X_batch,Y_batch) in mini_batches_list:
                self.forwardProp(X_batch)
                Alk = self.cash["A" + str(self.L + 1)]
                cost = self.computeCost(Alk, Y_batch)
                #print(cost)
                self.backProp(X_batch, Y_batch)
                self.updateParameters[self.optimizer]()

            #if (i % 10 == 0):
            print("iteration ", i, ":", cost)
            self.costs.append(cost)

    def predict(self,X):
        X = X.astype(float)
        X = X.T
        self.forwardProp(X)
        y = self.cash["A"+str(self.L+1)]
        #pred = np.zeros(y.shape)
        label = np.argmax(y,axis=0)
        #for (i,x) in enumerate(label):
            #pred[x,i] = 1
        return label

    def gradiantCheck(self,X,Y):
        (X, Y) = self.preprocessing(X, Y)
        dwij = np.zeros(self.parameters["W1"].shape)
        self.forwardProp(X)
        Al = self.cash["A"+str(self.L+1)]
        fwij = self.computeCost(Al, Y)
        self.backProp(X, Y)
        for i in range(self.hidden_layers[0]):
            for j in range(X.shape[0]):
                wij = self.parameters["W1"][i, j]
                wije = wij + 0.001
                self.parameters["W1"][i, j] = wije
                self.forwardProp(X)
                Al = self.cash["A" + str(self.L + 1)]
                fwije = self.computeCost(Al, Y)
                self.parameters["W1"][i, j] = wij
                dwij[i,j] = (fwij - fwije) / (wij - wije)
        dw11bp = self.paraderiv["dW1"]
        return dw11bp - dwij











