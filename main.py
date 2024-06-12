import os
import pickle
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import BBOB.bbobfunctions as BBOB
# import BBOB.cecfunctions
from BBOB.utils import getFitness,genOffset,setOffset,getOffset
import torch
from imports import *
from GLHF.problem import Problem
from GLHF.GLHFMODEL import GB_GLHF as GLHF
from GLHF.GLHFMODEL import GLHF_NO_GUMBEL
from utils import logExpResult
import argparse

class taskProblem(Problem):
    def __init__(self,fun=None,repaire=True,dim=None):
        super().__init__()
        self.fun=fun
        self.useRepaire=repaire
        self.dim=dim

    def repaire(self,x):
        xlbmask=torch.zeros_like(x,device=DEVICE)
        xlbmask[x<self.fun['xlb']]=1
        normalmask=1-xlbmask
        xlbmask=xlbmask*self.fun['xlb']
        x=normalmask*x+xlbmask

        xubmask=torch.zeros_like(x,device=DEVICE)
        xubmask[x>self.fun['xub']]=1
        normalmask=1-xubmask
        xubmask=xubmask*self.fun['xub']
        x=normalmask*x+xubmask
        return x
    
    def calfitness(self,x):
        '''
        input：x(batch,n,d)
        返回：（batch,n,d+1） , fitness(b,n,1)
        '''
        
        # x=torch.matmul(x,self.w) #b,n,d
        
        if self.useRepaire:
            x1=self.repaire(x)
        else:
            x1=x
            
            
        b,n,d=x.shape
        x1=x1.view((-1,d))
        r=getFitness(x1,self.fun)   #b,n,1
        r=torch.unsqueeze(r,-1)
        r=r.view((b,n,1))
        x1=x1.view((b,n,d))
        pop=torch.cat((r,x1),dim=-1) #b,n,d
        return pop,r
    
    
    def genRandomPop(self,batchShape):
        lb=self.fun['xlb'] 
        ub=self.fun['xub']
        return torch.rand(batchShape,device=DEVICE)*(ub-lb)+lb

    def reoffset(self):
        genOffset(self.dim,self.fun)
        
        
    def setOffset(self,offset):
        for key in offset.keys():
            self.fun[key]=offset[key]
    

    def getfunname(self):
        return self.fun['fid']
    
    def setfun(self,fun):
        self.fun=fun
        
        
# @logExpResult('./nogumbel_newarch5_BBOB实验d100.xlsx','sheet1')
def eval(step=100,model=None,popsize=100,problemdim=100,expname='轮盘赌',fid=1,offsets=None,runs=3):
    trails={
        'name':expname,
        'trail':[],
        'evalnum':[],
        'mean':[],
        'std':[]
    } 
    model=model.eval()
    for i in range(runs):
        trail=[]
        evalnum=[]
        fun=BBOB.FUNCTIONS[fid]
        fun['xlb']=-10
        fun['xub']=10 #设置上下界
        task=taskProblem(fun=fun,repaire=True,dim=problemdim)
        offset=offsets[fid]
        setOffset(fun,offset)
        task.setfun(fun)
        pop=task.genRandomPop((1,popsize,problemdim))
        pop,r=task.calfitness(pop)
        evallosslist=[]
        trail.append(torch.min(r).detach().cpu().numpy())
        evalnum.append(0)
        bar2=tqdm(range(step),ncols=100)
        plt.cla()
        
        for i in bar2:
            pop=pop.detach()
            pop2,_,_=model(pop,task)
            pop=pop2
            loss=torch.min(pop[...,0])
            evallosslist.append(loss.item())
            bar2.set_description(' %d eval inner loop %.6E'%(fid,torch.min(pop[...,0])))
            trail.append(torch.min(pop[...,0]).detach().cpu().numpy())
            evalnum.append(i+1)
        trails['trail'].append(trail)
        trails['evalnum'].append(evalnum)
        plt.plot(evallosslist)
        plt.savefig('imgs/%s_bbob_f%d.png'%(expname,fid))
        plt.cla()
    trails['trail']=np.array(trails['trail'])
    trails['mean']=np.array(trails['trail'])
    m=np.mean(trails['trail'][...,-1])
    s=np.mean(trails['trail'][...,-1])
    m='%.2E'%m
    s='%.2E'%s
    trails['mean']=m
    trails['std']=s
    #保存
    # with open('./trails/dim%d/%s(%s_dim%d).pkl'%(problemdim,expname,task.getfunname(),problemdim),'wb') as f:  
    #     pickle.dump(trails,f)
        
    print(fid,expname,'%s(%s)'%(m,s))
    return fid,expname,'%s(%s)'%(m,s)

    


def test(dim=100,mutdim=1000,step=100,crdim=4,popsize=100,ckpt='./ckpt/new_arch_4.pth'):
    model=GLHF(popsize=popsize,selmod='1-to-1',
               cr_policy='learned',muthdim=mutdim,crhdim=crdim).to(DEVICE)
    model.load_state_dict(torch.load(ckpt))
    model=model.to(DEVICE)
    with open('bbobOffsets_dim%d.pkl'%dim,'rb') as f:
        offsets=pickle.load(f)

    funidset=[i for i in range(1,25)]
    # funidset=[20]
    bar=tqdm(funidset)
    for fid in bar:
        with torch.no_grad():
            _,_,res=eval(step=step,model=model,popsize=popsize,problemdim=dim,expname='GLHF',fid=fid,offsets=offsets,runs=3)
            bar.set_description('F%d,%s'%(fid,res))








def parsearge():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-d', '--dim', choices=[30,100],type=int,default=30,help='dimention of the task.')
    parser.add_argument('-n', '--popsize',type=int,default=100,help='population size')
    parser.add_argument('-t', '--step',type=int,default=100,help='how many steps to evolve.')
    parser.add_argument('-p','--ckpt',type=str, default='./ckpt/pom_m_release.pth',help='path to the checkpoint file')
    parser.add_argument('-s','--modelsize',type=str,choices=['vs','s','m','l','vl','xl'],
                        default='m',help='this parameter defines the model size.')
    
    args = parser.parse_args() 
    return args




if __name__=='__main__':
    modeldict={
        'vs':[200,4],
        's':[500,4],
        'm':[1000,4],
        'l':[2000,20],
        'vl':[5000,50],
        'xl':[10000,100]
    }
    args=parsearge()
    if not args.modelsize=='m':
        print('At the moment we only open source the m model.')
        assert(0)
    mutdim,crdim=modeldict[args.modelsize]
    test(dim=args.dim,ckpt=args.ckpt,step=args.step,
         mutdim=mutdim,crdim=crdim,popsize=args.popsize)

