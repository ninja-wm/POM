import numpy as np
import torch
import BBOB.bbobfunctions as BBOB
from BBOB.utils import getFitness,genOffset
from GLHF.imports import DEVICE
import os
import pandas as pd

def sortIndiv(batchPop):
    '''
    作用：
    将一批种群中的个体按照 fitness维度的值来排序号
    
    输入：
    batchPop:一批种群，维度为（batchSize,dim+1,L*L）
    返回:
    排好序的（batch,dim+1,w,h的矩阵）
    '''
    b,d,w,h=batchPop.shape
    fitness=batchPop[:,0,:,:]
    fitness=fitness.view(b,w*h)
    _,fit=torch.sort(fitness,dim=1) #b,n
    batchPop=batchPop.view(b,d,-1).permute(0,2,1) #b,n,dim
    y=torch.zeros_like(batchPop)
    for index,pop in enumerate(batchPop):
        pop=batchPop[index]  #n,dim
        y[index]=torch.index_select(pop,0,fit[index])
    y=y.permute(0,2,1).view(b,d,w,h)
    batchPop=y
    return batchPop



def sortIndivBND(batchPop):
    b,n,d=batchPop.shape
    fitness=batchPop[...,0]
    _,fit=torch.sort(fitness,dim=1) #b,n
    y=torch.zeros_like(batchPop)
    for index,pop in enumerate(batchPop):
        pop=batchPop[index]  #n,dim
        y[index]=torch.index_select(pop,0,fit[index])
    batchPop=y
    return batchPop

def calFitness(batchChrom,fun):
    '''
    计算一个batch的种群的适应度
    返回（b,c+1,w,h）
    '''
    b,dim,w,h=batchChrom.shape
    batchChrom=batchChrom.view(b,dim,-1).permute(0,2,1)  #b,n,dim
    batchChrom=batchChrom.reshape(b*w*h,dim) #b*n,dim
    fitness=BBOB.getFitness(batchChrom,fun) #b,n
    batchChrom=batchChrom.view(b,w*h,dim)
    batchChrom=batchChrom.permute(0,2,1).view(b,dim,w,h) #b,c,w,h
    fitness=fitness.view(b,1,w,h)
    batchPop=[]
    for i in range(b):
        batchPop.append(torch.cat((fitness[i],batchChrom[i]),dim=0))
    batchPop=torch.stack(batchPop)
    return batchPop


def fitnessmap2vec(x,ranks):
    x=x.view(1,-1)  #1,w*h
    x=(x-torch.mean(x,dim=1))/torch.std(x,dim=1)
    x=x.view(-1,1)
    x=torch.cat((x,ranks),dim=1) #(w*h,2)
    return x


def logExpResult(filepath,sheetname):
    '''
    eg:
    'dim10','cecf4','CMA-ES',1.0
    
    '''
    if  not os.path.exists(filepath):
        pd.DataFrame({'dim':[10]*6+[100]*6,'F':['cecf%d'%i for i in range(4,10)]*2,'轮盘赌':['-']*12,'1-to-1':['-']*12,'learned':['-']*12}).to_excel(filepath,sheet_name=sheetname,index=False)
    
    linedict=dict()
    linedict['dim10']=dict()
    for i in range(0,6):
        linedict['dim10']['cecf%d'%(i+4)]=i
    linedict['dim100']=dict()
    for i in range(0,6):
        linedict['dim100']['cecf%d'%(i+4)]=i+6
            
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)  #输出的是函数名，维度和结果
            df=pd.read_excel(filepath,sheet_name=sheetname)
            dim,fname,expname,r=result
            row=linedict[dim][fname]
            df.loc[row,expname]='%.4E'%r
            df.to_excel(filepath,sheet_name=sheetname,index=False)
            print(df[['dim','F','轮盘赌','1-to-1','learned']])
            return result
        return wrapper
    return decorator    








