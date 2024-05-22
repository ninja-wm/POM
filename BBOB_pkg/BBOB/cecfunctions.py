
from BBOB.utils import *
################CEC####################


FUNCTIONS=dict()

def cecfun1(x,b=None,w=None):
    #(xi-bi)^2
    # b,n,dim
    #w (dim,1)
    batch,n,dim=x.shape
    z=x if b is None else x-b.view(-1)
    sc=torch.sin(z)
    sc=sc@w  #b,n,d @  d ,1 = b,n,1
    sc=torch.pow(sc,2).view(batch,n)
    return sc


FUNCTIONS['cecf1']={
'fid':'cecf1',
'fun':cecfun1,
'bias':None,
'w':None,
'xub':10,
'xlb':-10,
'bub':10,
'blb':-10,
}





def cecfun2(x,b=None):
    if not b is None:
        b=b.view(-1)
        z=x-b
    else:
        z=x
    sc=torch.sum(torch.abs(z),dim=2)
    return sc
    
    

FUNCTIONS['cecf2']={
'fid':'cecf2',
'fun':cecfun2,
'bias':None,
    'xub':10,
    'xlb':-10,
    'bub':10,
    'blb':-10,
}

def cecfun3(x,b=None):
    if not b is None:
        z=x-b
    else:
        z=x
    z1=z[:,:,:-1]
    z2=z[:,:,1:]
    sc=torch.sum(torch.abs(z1+z2),dim=2)+torch.sum(torch.abs(z),dim=2)
    return sc


FUNCTIONS['cecf3']={
'fid':'cecf3',
'fun':cecfun3,
'bias':None,
    'xub':10,
    'xlb':-10,
    'bub':10,
    'blb':-10,
}



def cecfun4(x,b=None):  #checked
    if not b is None:
        z=x-b
    else:
        z=x
    sc=torch.sum(torch.pow(z,2),dim=2)
    return sc    


FUNCTIONS['cecf4']={
'fid':'cecf4',
'fun':cecfun4,
'bias':None,
 'xub':100,
    'xlb':-100,
    'bub':50,
    'blb':-50,
}

def cecfun5(x,b=None):  #checked
    if not b is None:
        z=x-b
    else:
        z=x
    z=torch.abs(z)
    sc=torch.max(z,dim=2)[0]
    return sc

FUNCTIONS['cecf5']={
'fid':'cecf5',
'fun':cecfun5,
'bias':None,
 'xub':100,
    'xlb':-100,
    'bub':50,
    'blb':-50,
}

def cecfun6(x,b=None): #checked
    if not b is None:
        z=x-b
    else:
        z=x
    x1=z[:,:,:-1]
    x2=z[:,:,1:]
    return torch.sum(100*torch.pow((torch.pow(x1,2)-x2),2)+torch.pow((x1-1),2),dim=2)


FUNCTIONS['cecf6']={
'fid':'cecf6',
'fun':cecfun6,
'bias':None,
 'xub':100,
    'xlb':-100,
    'bub':50,
    'blb':-50,
}


def cecfun7(x,b=None):     #checked
    if not b is None:
        z=x-b
    else:
        z=x
    sc=torch.sum(torch.pow(z,torch.tensor(2).to(DEVICE))-10*torch.cos(2*np.pi*(z))+10,dim=2)
    return sc

FUNCTIONS['cecf7']={
'fid':'cecf7',
'fun':cecfun7,
'bias':None,
 'xub':5,
    'xlb':-5,
    'bub':2.5,
    'blb':-2.5,
}

def cecfun8(x,b=None):  #checked
    if not b is None:
        z=x-b
    else:
        z=x
    i=torch.from_numpy(np.array([i+1 for i in range(x.shape[2])])).view(-1).to(DEVICE).view(1,1,x.shape[2])
    sc=torch.sum(torch.pow(z,2)/4000,dim=2)-torch.prod(torch.cos((z)/torch.sqrt(i)),dim=2)+1
    return sc


FUNCTIONS['cecf8']={
'fid':'cecf8',
'fun':cecfun8,
'bias':None,
    'xub':600,
    'xlb':-600,
    'bub':300,
    'blb':-300,
}

def cecfun9(x,b=None):   #checked
    if not b is None:
        z=x-b
    else:
        z=x
    sc=-20*torch.exp(-0.2*torch.sqrt((1/x.shape[2])*torch.sum(torch.pow(z,2),dim=2))
                     )-torch.exp((1/x.shape[2])*torch.sum(torch.cos(2*np.pi*(z)),dim=2))+20+np.e
    return sc

FUNCTIONS['cecf9']={
'fid':'cecf9',
'fun':cecfun9,
'bias':None,
    'xub':32,
    'xlb':-32,
    'bub':16,
    'blb':-16,
}
    




    


    

    