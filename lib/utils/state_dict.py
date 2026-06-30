import torch
import os,sys


def copyRight(file1, file2, key1='box_head.conv5_size.weight', key2=None):
    stateDict_1 = torch.load(file1, map_location="cpu")['net']
    stateDict_2 = torch.load(file2, map_location="cpu")
    for key1 in list(stateDict_2):
        
        try:
            a = torch.prod(torch.tensor((stateDict_1[key1].shape)))
        except:
            print(key1, 'shape不一致')
            return 
        key2 = key1
        b = (stateDict_1[key1]==stateDict_2[key2]).sum()
        if a==b:
            
            continue
        else:
            print(key1, '参数不一致')


def delGrad(file):
    stateDict = torch.load(file, map_location="cpu")
    del stateDict['optimizer']
    torch.save(stateDict, file)


def qkv2q_k_v(file, new_file):
    
    stateDict = torch.load(file, map_location="cpu")['net']
    stateDict_new = {}
    for k,v in list(stateDict.items()):
        if 'qkv' in k:
            print(f'transfer \"{k}\".')
            q_key = k.replace('qkv','q_linear')
            k_key = k.replace('qkv','k_linear')
            v_key = k.replace('qkv','v_linear')
            if 'weight' in k:
                stateDict[q_key] = v[:768, :]
                stateDict[k_key] = v[768:768*2, :]
                stateDict[v_key] = v[768*2:, :]
            elif 'bias' in k:
                stateDict[q_key] = v[:768]
                stateDict[k_key] = v[768:768*2]
                stateDict[v_key] = v[768*2:]
            del stateDict[k]
        
        
    torch.save(stateDict, new_file)


def param_anl(file):
    
    stateDict = torch.load(file, map_location="cpu")['net']
    for k,v in list(stateDict.items()):
        if 'adapt' in k:
            print(f"{k}, \tmean={v.mean()}, \tstd={v.std()}\n")


if __name__=="__main__":
    sys.path.append(os.getcwd())
    
    
    
    
    
    delGrad(file='')
    
            
    
