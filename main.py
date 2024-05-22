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

    


def test(dim=100,nozero=False,ckpt='./ckpt/new_arch_4.pth'):
    psize=100
    model=GLHF(popsize=psize,selmod='1-to-1',cr_policy='learned',muthdim=1000,crhdim=4).to(DEVICE)
    model.load_state_dict(torch.load(ckpt))
    model=model.cuda()
    if not nozero:
        with open('bbobOffsets_dim%d.pkl'%dim,'rb') as f:
            offsets=pickle.load(f)
    else:
        with open('no_zero_bbobOffsets_dim%d.pkl'%dim,'rb') as f:
            offsets=pickle.load(f)
    # banlist=[3,5,12,15,17,18]
    # funidset=[i for i in range(1,25) if not i in banlist]
    # funidset=banlist
    funidset=[i for i in range(1,25)]
    # funidset=[20]
    bar=tqdm(funidset)
    for fid in bar:
        with torch.no_grad():
            _,_,res=eval(step=100,model=model,popsize=100,problemdim=dim,expname='GLHF',fid=fid,offsets=offsets,runs=3)
            bar.set_description('F%d,%s'%(fid,res))


def genBBOBoffset(dim=10):
    '''
    对所有bbob函数生成offset，并且把xopt和fopt设置为0，然后把offset保存到offsets.pkl
    '''
    offsets=dict()
    bar=tqdm(range(1,25))
    offsets=dict()
    for fid in bar :
        f=BBOB.FUNCTIONS[fid]
        genOffset(dim,f) 
        if not fid in [5,24]:
            f['xopt']=torch.zeros((dim,)).cuda()
        f['fopt']=0
        offsets[fid]=getOffset(f)
    with open('no_zero_bbobOffsets_dim%d.pkl'%dim,'wb') as f:
        pickle.dump(offsets,f)





dimention=100
if __name__=='__main__':
    # for i in [100,500,1000,10000]:
    #     genBBOBoffset(dim=i)
    # genBBOBoffset(dim=100)
    test(dim=dimention,nozero=False,ckpt='./ckpt/cr100_release_nf5.pth')
    # changeBBOBoffset(dim=30)
