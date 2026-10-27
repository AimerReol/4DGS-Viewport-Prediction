from torch import nn
import torch
from .memory_predictor import Predictor as MemoryPredictor

class Predictor(MemoryPredictor):
    def __init__(self,method):
        assert method in ['Baseline','MotionQuery','SceneQuery']
        super().__init__('TransformerMemory')
        self.method_name=method
        if method!='Baseline':
            self.memory.query_condition=nn.Linear(64,32,bias=False)
            nn.init.zeros_(self.memory.query_condition.weight)

    def run(self,d,ids,encoded=None):
        features=d['features']
        if self.no_gaussian:
            features=features*d['gaussian_keep'][None,None]
        history=d['x'][ids];geometry=d['geometry'][ids];target=d['target'][ids]
        user=self.history(history)[1][-1]
        base=self.history_head(user).reshape(-1,len(self.horizons),6)
        if encoded is None:encoded=self.encode_scene(features,d['valid'],d['endpoint_frames'],d['endpoint_times'])
        horizon=history.new_tensor([h/10 for h in self.horizons]).view(1,-1,1).expand(len(ids),-1,-1)
        geometry_embedding=self.geometry_embed(geometry)
        h,_=self.graph(encoded[d['contexts'][ids]],d['edges'][target],d['valid'][target],user,horizon,geometry_embedding,None)
        if self.has_memory:
            frames=d['past_frames'][ids];times=d['past_times'][ids]
            pf=features[frames]
            past=torch.cat([pf,d['past_geometry'][ids],times[:,:,None,:].expand(-1,-1,64,-1)],-1).permute(0,2,1,3)
            query=torch.cat([features[target],geometry[...,:7],horizon[:,:,None].expand(-1,-1,64,-1)],-1)
            condition=None
            if self.method_name=='MotionQuery':
                query_user=d['query_user_override'][ids] if 'query_user_override' in d else user
                condition=query_user[:,None,None,:].expand(-1,len(self.horizons),64,-1)
            elif self.method_name=='SceneQuery':
                condition=query[...,:64]
            if d.get('disable_query_condition',False):condition=None
            h=h+self.memory(past,history[:,d['past_offsets']],query,times,condition)
        values=torch.cat([h,geometry_embedding],-1)
        values=torch.cat([values,self.null_value.expand(len(ids),len(self.horizons),1,48)],2)
        available=torch.cat([d['valid'][target],torch.ones_like(target[...,None],dtype=torch.bool)],-1)
        logits=self.importance(torch.cat([user[:,None,None].expand(-1,len(self.horizons),65,-1),values],-1)).squeeze(-1)
        attention=logits.masked_fill(~available,-1e9).softmax(-1)
        context=(attention[...,None]*values).sum(2)
        residual=self.fusion(torch.cat([user[:,None].expand(-1,len(self.horizons),-1),context,horizon],-1))
        return dict(pose=base+residual,attention_position=attention,attention_orientation=attention)
