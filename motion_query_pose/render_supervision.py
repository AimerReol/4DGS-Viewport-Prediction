import json,math,sys
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .config import DATA,SOURCE,PROJECT,OLD,RENDER,FIELD_PATH

def quaternion_matrix(q):
 q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True).clamp_min(1e-12)
 w,x,y,z=q.unbind(-1)
 return torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1).reshape(*q.shape[:-1],3,3)

def rotvec_matrix(v):
 theta=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
 q=torch.cat([torch.cos(theta/2),v*(.5*torch.sinc(theta/(2*math.pi)))],-1)
 return quaternion_matrix(q)

def pack_cov(c):return c[..., [0,0,0,1,1,2],[0,1,2,1,2,2]].contiguous()
def unpack_cov(c):
 return torch.stack([c[...,0],c[...,1],c[...,2],c[...,1],c[...,3],c[...,4],c[...,2],c[...,4],c[...,5]],-1).reshape(*c.shape[:-1],3,3)

def make_cam(position,rotation,alignment,side,fov):
 a=np.asarray(alignment['matrix']);b=np.asarray(alignment['translation'])
 w=np.diag([1,-1,1])@rotation.T@a;c=a.T@(position-b)
 focal=side/(2*np.tan(np.deg2rad(fov/2)))
 return SimpleNamespace(R=w.astype(np.float32),t=(-w@c).astype(np.float32),c=c.astype(np.float32),K=np.array([[focal,0,(side-1)/2],[0,focal,(side-1)/2],[0,0,1]],np.float32),width=side,height=side)

class NativeRenderer:
 def __init__(self,side=None,fov=None):
  sys.path.insert(0,str(PROJECT));from aps import backend as be
  be.activate();self.be=be
  from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
  self.side=side or RENDER['side'];self.fov=fov or RENDER['fov_degrees']
  focal=self.side/(2*math.tan(math.radians(self.fov/2)))
  cam=SimpleNamespace(R=np.eye(3,dtype=np.float32),t=np.zeros(3,np.float32),c=np.zeros(3,np.float32),K=np.array([[focal,0,(self.side-1)/2],[0,focal,(self.side-1)/2],[0,0,1]],np.float32),width=self.side,height=self.side)
  self.cam=be.camera_view(cam,0)
  settings=GaussianRasterizationSettings(image_height=self.side,image_width=self.side,tanfovx=math.tan(self.cam.FoVx/2),tanfovy=math.tan(self.cam.FoVy/2),bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=self.cam.world_view_transform,projmatrix=self.cam.full_proj_transform,sh_degree=0,campos=self.cam.camera_center,prefiltered=False,bwd_depth=False,bwd_dynamic=False,debug=False)
  self.raster=GaussianRasterizer(settings)
  self.alignment=json.loads((DATA/'data_manifest.json').read_text())['alignment']
  self.a=torch.tensor(self.alignment['matrix'],dtype=torch.float32,device='cuda');self.b=torch.tensor(self.alignment['translation'],dtype=torch.float32,device='cuda')
  self.flip=torch.diag(torch.tensor([1.,-1.,1.],device='cuda'))
  self.field_array=np.load(FIELD_PATH,mmap_mode='r')
  index=np.load(DATA/'cache/render_index.npz');self.labels=torch.tensor(index['labels'],device='cuda');self.fake_depth=torch.tensor(index['fake_depth'],device='cuda')
  self.frame_map={int(t):i for i,t in enumerate(index['frames'])};self.lru=OrderedDict()

 def field(self,frame):
  frame=int(frame)
  if frame not in self.lru:
   self.lru[frame]=torch.tensor(np.array(self.field_array[self.frame_map[frame]],copy=True),device='cuda')
   if len(self.lru)>2:self.lru.popitem(last=False)
  self.lru.move_to_end(frame);return self.lru[frame]

 def geometry(self,field,position,rotation,variant):
  w=self.flip@rotation.T@self.a
  means=(field[:,:3]@self.a.T+self.b-position)@rotation@self.flip
  if variant=='RenderSphere':cov=torch.diag_embed(field[:,10,None].expand(-1,3))
  else:cov=unpack_cov(field[:,3:9])
  cov=w[None]@cov@w.T[None]
  if variant=='RenderDepthShuffle':

   depth=means[:,2];ratio=torch.where(depth>.2,self.fake_depth/depth.clamp_min(.2),torch.ones_like(depth))
   means=means*ratio[:,None];cov=cov*ratio[:,None,None].square()
  return means,pack_cov(cov)

 def render(self,frame,position,rotation,colors,variant='RenderFull'):
  field=self.field(frame);means,cov=self.geometry(field,position,rotation,variant)
  out=self.raster(means3D=means.contiguous(),means2D=torch.zeros_like(means),opacities=field[:,9:10].contiguous(),colors_precomp=colors.contiguous(),shs=None,scales=None,rotations=None,dynamics=torch.zeros_like(field[:,9:10]),cov3D_precomp=cov)
  return out[0]

 def relative(self,current_position,current_rotation,relative):
  return current_position+current_rotation@relative[:3],current_rotation@rotvec_matrix(relative[3:])

 def pair(self,frame,current_position,current_rotation,pred,true,probes,variant='RenderFull'):
  colors=probes[self.labels]
  pp,pr=self.relative(current_position,current_rotation,pred)
  with torch.no_grad():
   tp,tr=self.relative(current_position,current_rotation,true)
   target=self.render(frame,tp,tr,colors,variant)
  image=self.render(frame,pp,pr,colors,variant)
  return (image-target).square().mean()

 def anchor_maps(self,frame,position,rotation,variant='RenderFull'):
  images=[]
  for start in range(0,64,3):
   ids=torch.arange(start,start+3,device='cuda');colors=(self.labels[:,None]==ids[None]).float()
   images.append(self.render(frame,position,rotation,colors,variant))
  return torch.cat(images)[:64]

def probes(rng,channels=3):return torch.tensor(rng.choice([-1.,1.],size=(64,channels)),dtype=torch.float32,device='cuda')

class Supervision:
 def __init__(self,renderer,arrays):
  self.renderer=renderer;self.arrays=arrays
  poses=np.load(OLD/'cache/poses.npz')['poses'];current=poses[arrays['users'],arrays['origins']]
  self.current_position=torch.tensor(current[:,:3],dtype=torch.float32,device='cuda')
  self.current_rotation=torch.tensor(Rotation.from_quat(current[:,3:]).as_matrix(),dtype=torch.float32,device='cuda')
  self.denominator=None

 def auxiliary(self,predicted_raw,target_raw,chosen,rng,variant):
  selected=rng.choice(len(chosen),size=min(RENDER['samples_per_batch'],len(chosen)),replace=False)
  horizons=rng.integers(0,self.arrays['target'].shape[1],len(selected));terms=[];logged=[]
  for row,h in zip(selected,horizons):
   sample=int(chosen[row]);frame=int(self.arrays['target'][sample,h]);raw=predicted_raw[row,h]
   leaf=raw.detach().clone().requires_grad_(True);code=probes(rng)
   discrepancy=self.renderer.pair(frame,self.current_position[sample],self.current_rotation[sample],leaf,target_raw[row,h].detach(),code,variant)/self.denominator[h]
   gradient=torch.autograd.grad(discrepancy,leaf)[0]
   assert torch.isfinite(gradient).all() and torch.isfinite(discrepancy)

   tangent=(raw*gradient.detach()).sum()
   terms.append(tangent-tangent.detach()+discrepancy.detach());logged.append(float(discrepancy.detach()))
  return torch.stack(terms).mean(),float(np.mean(logged))
