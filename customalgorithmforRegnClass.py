import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF']) # red , green,  blue
cmap_light = ListedColormap(['#FFBBBB', '#BBFFBB', '#BBBBFF']) # light red to light blue

def sigmoid(h):
    return 1/(1+np.exp(-h))

def cross_entropy(y, p_hat):
    return -(1/len(y))*np.sum(y*np.log(p_hat)+ (1-y)*np.log(1-p_hat))

def accuracy(y, y_hat):
    return np.mean(y==y_hat)
def MinMaxScaler(X):
    result=((X-X.min())/(X.max()-X.min()))
    return result

class SLogisticRegression():
    
    def __init__(self, thresh=0.5):
        self.thresh=thresh
        self.w=None
        self.b=None
   
    def fit(self, X, y, eta=1e-3, epochs=1e3, show_curve=False):
        epochs=int(epochs)
        N,D = X.shape
        
        #initialize weights
        self.w = np.random.randn(D)
        self.b=np.random.randn(1)
        
        J=np.zeros(epochs)
        
        for epoch in range(epochs):
            p_hat=self.__forward__(X)
            J[epoch]= cross_entropy(y, p_hat)
            
            # update weights
            self.w -= eta*(1/N)*X.T@(p_hat-y)
            self.b-= eta*(1/N)*np.sum(p_hat-y)
            
            
        if show_curve:
            plt.figure()
            plt.plot(J)
            plt.xlabel('epochs')
            plt.ylabel('$\mathcal{J}$')
            plt.title('Training Curve')
    def __forward__(self, X):
        return sigmoid(X@self.w + self.b)
    def predict(self, X):
        return (self.__forward__(X)>=self.thresh).astype(np.int32)

    
def derivative(Z,a):
    if a==linear:
        return 1
    elif a==sigmoid:
        return Z*(1-Z)
    elif a==np.tanh:
        return 1-Z*Z
    elif a==ReLU:
        return (Z>0).astype(int)
    else:
        ValueError('Unknown Activation Function')
        
# functions        
def linear(H):
    return H

def ReLU(H):
    return H*(H>0)

def sigmoid(H):
    return 1/(1+np.exp(-H))
def softmax(H):
    eH=np.exp(H)
    return eH/eH.sum(axis=1, keepdims=True)

def one_hot_encode(y):
    N=len(y)
    K=len(set(y))
    Y=np.zeros((N,K))
    for i in range(N):
        Y[i,y[i]]=1
    return Y
#loss function
def cross_entropy(Y, P_hat):
    return -(1/len(Y))*np.sum(Y*np.log(P_hat))
def OLS(Y, Y_hat):
    return (1/(2*len(Y)))*np.sum((Y-Y_hat)**2)
def accuracy(y, y_hat):
    return np.mean(y==y_hat)
def R2(y, y_hat):
    return 1 - np.sum((y-y_hat)**2)/np.sum((y-y.mean())**2)

# Articial Neural Network
class ANN():
  def __init__(self, architecture, activations=None, mode=0):
    self.mode=mode
    self.architecture=architecture
    self.activations=activations
    self.L = len(architecture)+1

  def fit(self, X, y, eta=1e-3, epochs =1e3, show_curve=False):
    epochs=int(epochs)
    if self.mode:
      Y=y
    else:
      Y =one_hot_encode(y)

    N,D =X.shape
    K=Y.shape[1]

    #Initialize Weights and Biases:  Stochastic Initialization
    self.W ={l: np.random.randn(M[0],M[1]) for l, M in enumerate
             (zip(([D]+self.architecture), (self.architecture +[K])),1)}
    self.b = {l: np.random.randn(M) for l,M in enumerate(self.architecture +[K],1)}

    #Activation Function Loading
    if self.activations is None:
      self.a = {l: ReLU for l in range(1, self.L)}
    else:
      self.a = {l: act for l, act in enumerate(self.activations, 1)} 

    # Mode Output Activation Function Set
    if self.mode:
      self.a[self.L]= linear
    else:
      self.a[self.L] = softmax

    J = np.zeros(epochs)

    #Gradient Descent/ Back Prop

    for epoch in range(epochs):
      self.forward(X)
      if self.mode:
        J[epoch]= OLS(Y, self.Z[self.L])
      else:
        J[epoch]= cross_entropy(Y, self.Z[self.L])

      dH = (1/N)*(self.Z[self.L]-Y)

      for l in sorted(self.W.keys(), reverse= True):
        dW = self.Z[l-1].T@dH
        db = dH.sum(axis=0)

        self.W[l] -= eta*dW
        self.b[l] -= eta*db

        if l>1:
          dZ = dH@self.W[l].T
          dH = dZ*derivative(self.Z[l-1], self.a[l-1]) 
    
    if show_curve:
        plt.figure()
        plt.plot(J)
        plt.xlabel("epochs")
        plt.ylabel("$\mathcal{J}$")
        plt.show()

  def forward(self, X):
    self.Z ={0:X}
    for l in sorted(self.W.keys()):
      self.Z[l] = self.a[l](self.Z[l-1]@self.W[l]+ self.b[l]) 

  def predict ( self, X):
    self.forward(X)
    if self.mode:
      return self.Z[self.L]
    else:
      return self.Z[self.L].argmax(axis=1)

class KNNclassifier():
    def fit(self, X, y):
        self.X=X
        self.y=y
    def predict(self, X, k , epsilon=1e-3):
        N = len(X)   # number of rows
        y_hat = np.zeros(N)
        for i in range(N):
              dist2 = np.sum((self.X-X[i])**2, axis=1) 
              idxt = np.argsort(dist2)[:k]
              gamma_k =1/np.sqrt(dist2[idxt]+epsilon)
              y_hat[i] =np.bincount(self.y[idxt], weights=gamma_k).argmax()
        return y_hat
