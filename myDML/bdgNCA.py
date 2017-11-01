import numpy as np
import math
class bdgNCA:
    def __init__(self,traindata):
        self.rawData=traindata[0]*0.1
        self.label=traindata[1]
        #self.A=np.random.rand(self.rawData.shape[1],self.rawData.shape[1])
        self.A = np.eye(self.rawData.shape[1], self.rawData.shape[1])
        self.label2indexs=self.classifyLabel()
        self.startPoint=np.zeros(self.rawData.shape[1])
        self.endPoint = np.zeros(self.rawData.shape[1])
        self.normlization()

#test_myDML itr=1000,k=20,rate=0.05, 0.015,0.015
#letter_recognition itr=200,k=20,rate=0.05
    def train(self,itr=200,k=20,rate=0.05):
        for i in range(itr):
            print(i)
            self.A+=self.deltaA(k)*rate

    def normlization(self):
        for i in range(len(self.startPoint)):
            self.startPoint[i]=min(self.rawData[:,i])
            self.endPoint[i] = max(self.rawData[:,i])
        for i in range(len(self.label)):
            for j in range(len(self.startPoint)):
                self.rawData[i,j]=(self.rawData[i,j]-self.startPoint[j])/(self.endPoint[j]-self.startPoint[j])


    def deltaA(self,k):
        l=len(self.label)
        randomSample=np.random.random_integers(l-1,size=k)
        s=np.zeros(self.A.shape)
        for i in randomSample:
            s1 = np.zeros(s.shape)
            s2= np.zeros(s.shape)
            norm=0
            Pi=np.zeros(l)
            Ci=0
            for k in range(l):
                if i==k:
                    Pi[k]=0
                else:
                    Pi[k]=math.exp(-math.pow(self.myDistance(self.rawData[i],self.rawData[k]),2))
                #print(i,k,Pi[k],self.myDistance(self.rawData[i],self.rawData[k]),self.Euclidean_distance(self.rawData[i],self.rawData[k]))
                norm+=Pi[k]
            for k in range(l):
                xik = self.rawData[i] - self.rawData[k]
                Pi[k]/=norm
                s1+=Pi[k]*np.dot(np.transpose([xik]),[xik])
            for j in self.label2indexs[self.label[i]]:
                xij = self.rawData[i] - self.rawData[j]
                Ci+=Pi[j]
                s2+=Pi[j]*np.dot(np.transpose([xij]),[xij])
            s+=s1-s2/Ci
        return 2*np.dot(self.A,s)

    def Euclidean_distance(self,inst_a, inst_b):
        return np.linalg.norm(inst_a - inst_b)

    def myDistance(self,vec_a,vec_b):
        a=[(vec_a[i]-self.startPoint[i])/(self.endPoint[i]-self.startPoint[i]) for i in range(len(vec_a))]
        b = [(vec_b[i] - self.startPoint[i]) / (self.endPoint[i] - self.startPoint[i]) for i in range(len(vec_a))]
        '''for i in range(len(vec_a)):
            vec_a[i]=(vec_a[i]-self.startPoint[i])/(self.endPoint[i]-self.startPoint[i])
            vec_b[i] =(vec_b[i] - self.startPoint[i]) / (self.endPoint[i] - self.startPoint[i])'''
        trans_a=np.dot(self.A,a)
        trans_b=np.dot(self.A,b)
        return np.linalg.norm(trans_a-trans_b)

    def classifyLabel(self):
        label2indexs = {}
        for i in range(len(self.label)):
            if self.label[i] in label2indexs.keys():
                label2indexs[self.label[i]].append(i)
            else:
                label2indexs[self.label[i]] = [i]
        return label2indexs