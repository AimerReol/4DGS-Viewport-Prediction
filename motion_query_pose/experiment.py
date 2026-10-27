import json,subprocess,time
import numpy as np
from .settings import *
def status(stage,**kw):save_json(ROOT/'status.json',dict(stage=stage,updated_at=time.time(),**kw))
def snapshot():
 line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]
 used,free,util=map(int,line.split(','));return dict(memory_used_mib=used,memory_free_mib=free,gpu_utilization_percent=util)
def wait_for_resources(stage):
 import os
 output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 external=[int(x.strip()) for x in output.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
 if external:raise RuntimeError(f'External CUDA processes present: {external}; retry when the GPU is available.')

def runtime():
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False
 torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.65)

def tensors(split,device='cuda'):
 import torch
 a=dict(np.load(DATA/'cache'/f'{split}_inputs.npz'));s=dict(np.load(PARENT/'cache/scene.npz'));d=dict(arrays=a,split=split)
 slots=np.arange(50).reshape(5,10)[:,np.rint(np.linspace(0,9,6)).astype(int)].reshape(-1)
 frames=a['origins'][:,None]-49+slots;times=np.broadcast_to(((slots-49)/10)[None,:,None],(len(frames),30,1)).copy().astype(np.float32)
 for k in ['x','geometry','y','target','contexts','y_scale','loss_scales']:d[k]=torch.as_tensor(a[k],device=device)
 for k in ['features','edges','valid','endpoint_frames','endpoint_times']:d[k]=torch.as_tensor(s[k],device=device)
 for k,v in dict(past_frames=frames,past_offsets=slots,past_times=times,past_geometry=a['past_geometry'],gaussian_keep=np.ones(65,np.float32)).items():d[k]=torch.as_tensor(v,device=device)
 return d
def encode(model,d):return model.encode_scene(d['features'],d['valid'],d['endpoint_frames'],d['endpoint_times'])
def forward(model,d,ids,encoded=None):return model.run(d,ids,encoded)
