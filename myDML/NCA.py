import numpy as np
import math
class NCA:
    def __init__(self,traindata):
        self.rawData=traindata[0]
        self.label=traindata[1]
        self.A=np.random.rand(self.rawData.shape[1],self.rawData.shape[1])
        self.distanceMatrix=np.zeros((len(self.label),len(self.label)))
        self.rowDistanceSum=np.zeros(len(self.label))
        self.label2indexs=self.classifyLabel()

    def train(self,itr=50,learnRate=0.02):
        for i in range(itr):
            print(i)
            self.updateDistanceMatrix()
            self.A+=learnRate*self.deltaA()

    def myDistance(self,vec_a,vec_b):
        trans_a=np.dot(self.A,vec_a)
        trans_b=np.dot(self.A,vec_b)
        return np.linalg.norm(trans_a-trans_b)

    def classifyLabel(self):
        label2indexs = {}
        for i in range(len(self.label)):
            if self.label[i] in label2indexs.keys():
                label2indexs[self.label[i]].append(i)
            else:
                label2indexs[self.label[i]] = [i]
        return label2indexs

    def updateDistanceMatrix(self):
        l=len(self.label)
        self.rowDistanceSum = np.zeros(l)
        for i in range(l):
            for j in range(i+1):
                self.distanceMatrix[i,j]=math.exp(-math.pow(self.myDistance(self.rawData[i],self.rawData[j]),2))
                self.distanceMatrix[j,i]=self.distanceMatrix[i,j]
        for i in range(l):
            #s=sum(self.distanceMatrix[i,:])
            for j in range(l):
                #self.distanceMatrix[i,j]/=s
                if i!=j:
                    self.rowDistanceSum[i]+=self.distanceMatrix[i,j]

    def Pij(self,i,j):
        if i==j:
            return 0
        return self.distanceMatrix[i,j]/self.rowDistanceSum[i]

    def Pi(self,i):
        p=0
        indexs=self.label2indexs[self.label[i]]
        for j in indexs:
            p+=self.Pij(i,j)
        return p

    def deltaA(self):
        s=np.zeros(self.A.shape)
        l=len(self.label)
        for i in range(l):
            s1 = np.zeros(self.A.shape)
            s2= np.zeros(self.A.shape)
            for k in range(l):
                xik=self.rawData[i]-self.rawData[k]
                s1+=self.Pij(i,k)*np.dot(np.transpose([xik]),[xik])
            indexs=self.label2indexs[self.label[i]]
            for j in indexs:
                xij=self.rawData[i]-self.rawData[j]
                s2+=self.Pij(i,j)*np.dot(np.transpose([xij]),[xij])
            s+=self.Pi(i)*s1-s2
        return 2*np.dot(self.A,s)

    def fA(self):
        f=0
        for i in range(len(self.label)):
           f+=self.Pi(i)
        return f