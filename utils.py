import os
import pandas as pd
import argparse


def argsParser():
    parser = argparse.ArgumentParser(description='Please tell me the Dimention')
    parser.add_argument('--dim','-dim',required=True, type=int, help='dimension')
    return parser.parse_args()
    

def logExpResult2(filepath,sheetname):
    
    if  not os.path.exists(filepath):
        pd.DataFrame({'F':['F%d'%i for i in range(1,25)],'B2Opt':['-']*24,'ES':['-']*24,'DE':['-']*24,'CMA-ES':['-']*24,'I-POP-CMA-ES':['-']*24,'LSHADE':['-']*24,'LES':['-']*24,'LGA':['-']*24,}).to_excel(filepath,sheet_name=sheetname,index=False)

    linedict=dict()
    for i in range(24):
        linedict[i+1]=i

    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)  #输出的是函数名，维度和结果
            df=pd.read_excel(filepath,sheet_name=sheetname)
            fname,expname,r=result
            row=linedict[fname]
            df.loc[row,expname]=r
            df.to_excel(filepath,sheet_name=sheetname,index=False)
            print(df[['F','B2Opt','ES','DE','CMA-ES','I-POP-CMA-ES','LSHADE','LES','LGA']])
            return result
        return wrapper
    return decorator    

##4
def logExpResult(filepath,sheetname):
    
    if  not os.path.exists(filepath):
        pd.DataFrame({'F':['F%d'%i for i in range(1,25)],'GLHF':['-']*24,'ES':['-']*24,'DE':['-']*24,'CMA-ES':['-']*24,'LSHADE':['-']*24,'LES':['-']*24,'LGA':['-']*24,}).to_excel(filepath,sheet_name=sheetname,index=False)

    linedict=dict()
    for i in range(24):
        linedict[i+1]=i

    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)  #输出的是函数名，维度和结果
            df=pd.read_excel(filepath,sheet_name=sheetname)
            fname,expname,r=result
            row=linedict[fname]
            df.loc[row,expname]=r
            df.to_excel(filepath,sheet_name=sheetname,index=False)
            print(df[['F','GLHF','ES','DE','CMA-ES','LSHADE','LES','LGA']])
            return result
        return wrapper
    return decorator    
