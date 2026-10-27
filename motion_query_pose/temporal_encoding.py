import math
import torch
from torch import nn

class TransformerMixer(nn.Module):
    def __init__(self,dim):
        super().__init__();self.attention=nn.MultiheadAttention(dim,4,dropout=0,batch_first=True)
    def forward(self,x):
        mask=torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).triu(1)
        return self.attention(x,x,x,attn_mask=mask,need_weights=False)[0]

class Sequence(nn.Module):
    def __init__(self,kind,input_dim,dim):
        super().__init__();self.project=nn.Linear(input_dim,dim) if input_dim!=dim else nn.Identity()
        self.norm1=nn.LayerNorm(dim);self.norm2=nn.LayerNorm(dim)
        assert kind=='Transformer', 'This export supports the frozen Transformer only'
        self.mix=TransformerMixer(dim)
        self.ff=nn.Sequential(nn.Linear(dim,2*dim),nn.GELU(),nn.Linear(2*dim,dim))
        self.final=nn.LayerNorm(dim)

        times=torch.arange(-149,1)[:,None];freq=torch.exp(torch.arange(0,dim,2)*(-math.log(10000)/dim))
        pos=torch.zeros(150,dim);pos[:,0::2]=torch.sin(times*freq);pos[:,1::2]=torch.cos(times*freq)
        self.register_buffer('positions',pos,persistent=False)
    def forward(self,x):
        assert 0<x.shape[1]<=150
        x=self.project(x)+self.positions[-x.shape[1]:].to(x)[None]
        x=x+self.mix(self.norm1(x));x=x+self.ff(self.norm2(x));x=self.final(x)
        return x,x[:,-1][None]

class AnchorMemory(nn.Module):
    def __init__(self,kind,correct=True):
        super().__init__();self.kind=kind;self.correct=correct
        self.keys=nn.Sequential(nn.Linear(73,32),nn.Tanh())
        self.values=nn.Sequential(nn.Linear(9,32),nn.Tanh())
        self.queries=nn.Sequential(nn.Linear(73,32),nn.Tanh())
        self.time=nn.Linear(1,32,bias=False)
        assert kind=='Transformer'
        self.sequence=Sequence(kind,32,32)
        self.read=nn.MultiheadAttention(32,4,batch_first=True)
        self.output=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,32))
        nn.init.normal_(self.output[-1].weight,0,.002);nn.init.zeros_(self.output[-1].bias)
    def forward(self,past,pose,query,times,condition=None):

        b,k,t,_=past.shape;h=query.shape[1]
        key=self.keys(past).reshape(b*k,t,32)
        value=self.values(pose)[:,None].expand(-1,k,-1,-1).reshape(b*k,t,32)
        time=self.time(times[:,None].expand(-1,k,-1,-1).reshape(b*k,t,1))
        q=self.queries(query) if condition is None else self.queries[1](self.queries[0](query)+self.query_condition(condition))
        q=q.permute(0,2,1,3).reshape(b*k,h,32)
        sequence,_=self.sequence(key+value+time)
        y=self.read(q,sequence,sequence,need_weights=False)[0]
        return self.output(y).reshape(b,k,h,32).permute(0,2,1,3)
