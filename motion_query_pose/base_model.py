"""Factorial graph ablation: past-head conditioning x explicit motion relations."""
import torch
from torch import nn
from .previous_model import pose_loss

METHODS=['OriginalNoGraph','OriginalEdge','EndpointTimeEdge','IntervalNoGraph','IntervalEdge','ShuffledIntervalEdge']


class PosePredictor(nn.Module):
    def __init__(self,method,horizons=(2,5,10,20,30,50)):
        super().__init__()
        assert method in METHODS
        self.variant=method
        self.horizons=list(horizons)
        self.with_time=method not in ['OriginalNoGraph','OriginalEdge']
        method='NoGraph' if method in ['OriginalNoGraph','IntervalNoGraph'] else 'EdgeGraph'
        self.method=method
        self.head_condition=method in ['HeadGraph','HeadMotionGraph']
        self.motion_relation=method in ['MotionGraph','HeadMotionGraph']
        # Shared paths are constructed first for identical initial states per seed.
        self.history=nn.GRU(9,64,batch_first=True)
        self.history_head=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,6*len(self.horizons)))
        self.scene_embed=nn.Linear(65,32)
        self.scene_time=nn.GRU(32,32,batch_first=True)
        self.geometry_embed=nn.Sequential(nn.Linear(10,16),nn.ReLU())
        self.null_value=nn.Parameter(torch.zeros(48))
        self.importance=nn.Sequential(nn.Linear(112,48),nn.Tanh(),nn.Linear(48,1))
        self.fusion=nn.Sequential(nn.Linear(113,64),nn.ReLU(),nn.Linear(64,6))
        nn.init.normal_(self.fusion[-1].weight,0,.002);nn.init.zeros_(self.fusion[-1].bias)
        self.norms=nn.ModuleList([nn.LayerNorm(32) for _ in range(2)])
        if method=='NoGraph':
            # Similar active parameter count to the most expressive graph variant.
            self.local=nn.ModuleList([nn.Sequential(nn.Linear(32,110),nn.ReLU(),nn.Linear(110,32)) for _ in range(2)])
        elif method=='MeanGraph':
            self.local=nn.ModuleList([nn.Sequential(nn.Linear(96,55),nn.ReLU(),nn.Linear(55,32)) for _ in range(2)])
        else:
            extra=8 if self.motion_relation else 0
            self.messages=nn.ModuleList([nn.Sequential(nn.Linear(64+extra,32),nn.ReLU(),nn.Linear(32,32)) for _ in range(2)])
            self.scores=nn.ModuleList([nn.Sequential(nn.Linear(64+extra,32),nn.Tanh(),nn.Linear(32,1)) for _ in range(2)])
            if self.head_condition:
                self.query=nn.Sequential(nn.Linear(81,16),nn.Tanh())
                self.keys=nn.ModuleList([nn.Linear(64+extra,16) for _ in range(2)])
            if self.motion_relation:self.relation=nn.Sequential(nn.Linear(14,8),nn.Tanh())

        if self.with_time:
            self.time_embed=nn.Linear(1,32,bias=False)
            nn.init.zeros_(self.time_embed.weight)

    def encode_scene(self,features,valid,frames=None,time_offsets=None):
        t,k=features.shape[:2]
        h=torch.relu(self.scene_embed(features))*valid[...,None]
        if frames is None:
            frames=(torch.arange(t,device=h.device)[:,None]-torch.arange(5,-1,-1,device=h.device)).clamp_min(0)
        tokens=h[frames]
        if self.with_time:
            assert time_offsets is not None
            tokens=(tokens+self.time_embed(time_offsets[...,None])[:,:,None,:])*valid[frames][...,None]
        seq=tokens.permute(0,2,1,3).reshape(len(frames)*k,6,32)
        # Anchor/context sequences are independent along this batch dimension.
        # Chunk execution avoids the torch 2.4 CUDA attention grid limit for
        # 1765 contexts x 64 anchors, retaining every six-token sequence.
        hidden=torch.cat([self.scene_time(part)[1][-1] for part in seq.split(8192)],dim=0)
        return hidden.reshape(len(frames),k,32)*valid[frames[:,-1]][...,None]

    def graph(self,h,edges,valid,user,horizon,geometry_embedding,relations,return_edges=False):
        # Graph comes AFTER shared temporal encoding in every v2 variant.
        b,s,k,_=h.shape
        h=h.reshape(b*s,k,32);valid=valid.reshape(b*s,k);edges=edges.reshape(b*s,k,6)
        batch=torch.arange(b*s,device=h.device)[:,None,None]
        neighbor_valid=valid[batch,edges]
        query=self.query(torch.cat([user[:,None,None,:].expand(-1,s,k,-1),horizon[:,:,None,:].expand(-1,-1,k,-1),geometry_embedding],-1)) if self.head_condition else None
        if query is not None:query=query.reshape(b*s,k,1,16).expand(-1,-1,6,-1)
        rel=self.relation(relations).reshape(b*s,k,6,8) if self.motion_relation else None
        weights_out=[]
        for i,norm in enumerate(self.norms):
            if self.method=='NoGraph':
                update=self.local[i](h)
            else:
                neighbors=h[batch,edges]
                if self.method=='MeanGraph':
                    weights=neighbor_valid.float()/neighbor_valid.sum(-1,keepdim=True).clamp_min(1)
                    avg=(neighbors*weights[...,None]).sum(2)
                    update=self.local[i](torch.cat([h,avg,h-avg],-1))
                else:
                    pairs=torch.cat([h[:,:,None,:].expand_as(neighbors),neighbors-h[:,:,None,:]],-1)
                    message_input=torch.cat([pairs,rel],-1) if rel is not None else pairs
                    logits=self.scores[i](message_input).squeeze(-1)
                    if query is not None:
                        logits=logits+(self.keys[i](message_input)*query).sum(-1)/4
                    weights=logits.masked_fill(~neighbor_valid,-1e9).softmax(-1)*neighbor_valid
                    weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
                    update=(self.messages[i](message_input)*weights[...,None]).sum(2)
                if return_edges:weights_out.append(weights.reshape(b,s,k,6))
            h=torch.relu(norm(h+update))*valid[...,None]
        return h.reshape(b,s,k,32),weights_out

    def forward(self,history,geometry,target_frames,features,edges,valid,relations,encoded_scene=None,return_edges=False,graph_query_override=None,context_ids=None,sample_frames=None,time_offsets=None):
        user=self.history(history)[1][-1]
        base=self.history_head(user).reshape(-1,len(self.horizons),6)
        if encoded_scene is None:encoded_scene=self.encode_scene(features,valid,sample_frames,time_offsets)
        horizon=history.new_tensor([h/10 for h in self.horizons]).view(1,-1,1).expand(len(history),-1,-1)
        graph_user=user if graph_query_override is None else graph_query_override
        geometry_embedding=self.geometry_embed(geometry)
        h,edge_weights=self.graph(encoded_scene[target_frames if context_ids is None else context_ids],edges[target_frames],valid[target_frames],graph_user,horizon,geometry_embedding,relations[target_frames] if relations is not None else None,return_edges)
        values=torch.cat([h,geometry_embedding],-1)
        values=torch.cat([values,self.null_value.expand(len(history),len(self.horizons),1,48)],2)
        available=torch.cat([valid[target_frames],torch.ones_like(target_frames[...,None],dtype=torch.bool)],-1)
        u=user[:,None,None,:].expand(-1,len(self.horizons),values.shape[2],-1)
        logits=self.importance(torch.cat([u,values],-1)).squeeze(-1)
        attention=logits.masked_fill(~available,-1e9).softmax(-1)
        context=(attention[...,None]*values).sum(2)
        residual=self.fusion(torch.cat([user[:,None,:].expand(-1,len(self.horizons),-1),context,horizon],-1))
        return dict(pose=base+residual,attention=attention,edge_weights=edge_weights)
