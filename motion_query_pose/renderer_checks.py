import copy,json,time,traceback
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .pipeline import *
from .render_supervision import NativeRenderer,Supervision,probes,rotvec_matrix,quaternion_matrix,pack_cov,unpack_cov,make_cam
from .predictor import Predictor
from .pose_loss import pose_loss

def main():
 runtime();wait_for_resources('preflight');r=NativeRenderer();rng=np.random.default_rng(2026101220);details={};began=time.time()
 a=dict(np.load(DATA/'cache/R1_inputs.npz'));sup=Supervision(r,a);h=len(HORIZONS)-1

 candidates=[int(a['split_train'][53])]
 for user in np.unique(a['users'][a['split_train']]):
  selected=a['split_train'][a['users'][a['split_train']]==user]
  candidates.extend(selected[np.rint(np.linspace(0,len(selected)-1,3)).astype(int)].tolist())
 for i in candidates:
  frame=int(a['target'][i,h]);p=sup.current_position[i];rot=sup.current_rotation[i]
  with torch.no_grad():
   tp,tr=r.relative(p,rot,torch.tensor(a['y'][i,h],device='cuda'))
   alpha=r.render(frame,tp,tr,torch.ones((len(r.labels),3),device='cuda'))
   coverage=float((alpha[0]>.01).float().mean())
  if coverage>4/(r.side*r.side):break
 else:raise AssertionError('No nonempty training view for renderer preflight; inspect alignment')
 del alpha
 details['numerical_check_window']=dict(index=int(i),recording=int(a['users'][i]+1),origin=int(a['origins'][i]),target_frame=frame,foreground_coverage=coverage,training_only=True,all_experiment_windows_retained=True)
 truth=torch.tensor(a['y'][i,h],device='cuda');estimate=truth+torch.tensor([.02,-.01,.01,.03,-.04,.02],device='cuda');code=probes(rng)

 be=r.be;cfg=json.loads((SOURCE/'config.json').read_text());model,_=be.restore_model(cfg,be.load_torch(SOURCE/'base/model.pt')['model'])
 for param in be.all_params(model):param.requires_grad_(False)
 with torch.no_grad():
  raw=be.deformed(model,json.loads((DATA/'data_manifest.json').read_text())['reconstruction_times'][frame])
  cam=make_cam(p.cpu().numpy(),rot.cpu().numpy(),r.alignment,r.side,r.fov)
  original=be.packed_render(raw,be.camera_view(cam,json.loads((DATA/'data_manifest.json').read_text())['reconstruction_times'][frame]),degree=model.active_sh_degree,colors=code[r.labels],black=True)['render']
  transformed=r.render(frame,p,rot,code[r.labels]);error=(transformed-original).abs()
  details['native_equivalence']=dict(max_abs=float(error.max()),mean_abs=float(error.mean()))
  assert float(error.mean())<3e-5 and float(error.max())<.003,details
  sphere=(2*raw[1].mean(-1)).exp().double().pow(3);volume=(2*raw[1].double()).sum(-1).exp()
  details['sphere_volume_max_relative']=float(((sphere-volume).abs()/volume.clamp_min(1e-30)).max());assert details['sphere_volume_max_relative']<1e-4
 del model,raw,original,transformed,error;torch.cuda.empty_cache()

 with torch.no_grad():
  pp,pr=r.relative(p,rot,estimate);tp,tr=r.relative(p,rot,truth)
  pm=r.anchor_maps(frame,pp,pr);tm=r.anchor_maps(frame,tp,tr);delta=pm-tm
  signed=r.render(frame,pp,pr,code[r.labels]);linear=torch.einsum('kc,khw->chw',code,pm)
  details['signed_color_linearity_max_abs']=float((signed-linear).abs().max());assert details['signed_color_linearity_max_abs']<3e-5
  exact=delta.square().sum(0).mean();trial=[]
  for _ in range(512):trial.append(float(torch.einsum('kc,khw->chw',probes(rng),delta).square().mean()))
  details['stochastic_loss']=dict(exact=float(exact),estimate_mean=float(np.mean(trial)),standard_error=float(np.std(trial,ddof=1)/np.sqrt(len(trial))))
  assert abs(np.mean(trial)-float(exact))<5*np.std(trial)/np.sqrt(len(trial))
  zero=r.pair(frame,p,rot,truth,truth,code);assert float(zero)==0
  details['identical_pose_loss']=float(zero)
  field=r.field(frame);m,c=r.geometry(field,p,rot,'RenderFull');ms,cs=r.geometry(field,p,rot,'RenderDepthShuffle');good=m[:,2]>.2
  def projection(m,c):
   m=m[good];c=unpack_cov(c[good]);z=m[:,2];j=torch.zeros((len(m),2,3),device='cuda');j[:,0,0]=1/z;j[:,1,1]=1/z;j[:,0,2]=-m[:,0]/z.square();j[:,1,2]=-m[:,1]/z.square()
   return m[:,:2]/z[:,None],j@c@j.transpose(-1,-2)
  uv,cov=projection(m,c);uvs,covs=projection(ms,cs)
  details['depth_control']=dict(projected_center_max_abs=float((uv-uvs).abs().max()),projected_cov_relative_norm=float(torch.linalg.vector_norm(cov-covs)/torch.linalg.vector_norm(cov)),changed_order=True)
  assert details['depth_control']['projected_center_max_abs']<1e-4 and details['depth_control']['projected_cov_relative_norm']<1e-5

 pc=torch.tensor([[-.45,-.2,2.3],[.2,.4,2.8],[.5,-.1,3.2],[-.1,.15,3.8]],device='cuda')
 toy=torch.zeros((4,11),device='cuda');toy[:,:3]=(pc@r.flip-r.b)@r.a
 cov=torch.diag_embed(torch.tensor([[.03,.015,.01],[.015,.045,.02],[.025,.012,.016],[.03,.02,.01]],device='cuda'))
 rawcov=r.a.T[None]@r.flip[None]@cov@r.flip[None]@r.a[None];toy[:,3:9]=pack_cov(rawcov);toy[:,9]=torch.tensor([.6,.7,.5,.4],device='cuda');toy[:,10]=.02
 r.lru[-1]=toy;colors=torch.tensor([[1.,-.3,.5],[-.5,1.,.2],[.2,.1,1.],[-.6,.4,-.2]],device='cuda')
 target_r=torch.tensor([.02,-.03,.01,.025,.035,-.02],device='cuda')
 with torch.no_grad():target_image=r.render(-1,target_r[:3],rotvec_matrix(target_r[3:]),colors)
 def toy_loss(v):return (r.render(-1,v[:3],rotvec_matrix(v[3:]),colors)-target_image).square().mean()
 x=torch.tensor([-.01,.01,-.015,-.01,.015,.005],device='cuda',requires_grad=True);g=torch.autograd.grad(toy_loss(x),x)[0].detach();fd=[]
 for j in range(6):
  d=torch.zeros(6,device='cuda');d[j]=.001
  with torch.no_grad():fd.append(float((toy_loss(x+d)-toy_loss(x-d))/(2*.001)))
 fd=np.array(fd);gn=g.cpu().numpy();relative=np.linalg.norm(fd-gn)/max(np.linalg.norm(fd),1e-8)
 details['finite_difference']=dict(analytic=gn.tolist(),numerical=fd.tolist(),relative_norm_error=float(relative));assert relative<.06,details['finite_difference']
 r.lru.clear()

 gradients={}
 for variant in METHODS[1:]:
  x=estimate.detach().clone().requires_grad_(True);loss=r.pair(frame,p,rot,x,truth,code,variant);g=torch.autograd.grad(loss,x)[0]
  assert torch.isfinite(g).all() and float(g[:3].norm())>0 and float(g[3:].norm())>0
  gradients[variant]=dict(loss=float(loss.detach()),translation_gradient_norm=float(g[:3].norm()),rotation_gradient_norm=float(g[3:].norm()))
 details['real_scene_pose_gradients']=gradients

 d=tensors('R1');ids=d['arrays']['split_train'][:CONFIG['batch']];torch.manual_seed(12);reference=Predictor().cuda();ref=copy.deepcopy(reference.state_dict())
 for method in METHODS:
  torch.manual_seed(12);m=Predictor(method).cuda()
  for k,v in ref.items():torch.testing.assert_close(m.state_dict()[k],v,atol=0,rtol=0)
  del m
 reference.eval()
 with torch.no_grad():
  first=forward(reference,d,ids)['pose'];changed=dict(d,y=d['y']+999)
  torch.testing.assert_close(first,forward(reference,changed,ids)['pose'],atol=0,rtol=0)
 assert first.shape==(len(ids),len(HORIZONS),6)
 reference.train();pred=forward(reference,d,ids)['pose'];sup.denominator=torch.ones(len(HORIZONS),device='cuda')
 aux,_=sup.auxiliary(pred*d['y_scale'],d['y'][ids],ids,np.random.default_rng(5),'RenderFull')
 total=pose_loss(pred,d['y'][ids],d['y_scale'],d['loss_scales'])+.1*aux;total.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in reference.parameters())
 details['parameters']=sum(p.numel() for p in reference.parameters());details['all_active_gradients_finite']=True;details['labels_do_not_change_forward']=True;details['identical_initial_states']=True
 details['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30;details['seconds']=time.time()-began
 save_json(ROOT/'ours_preflight.json',dict(complete=True,data_manifest_sha256=sha(DATA/'data_manifest.json'),**details));status('preflight_passed');print('PREFLIGHT_PASSED',json.dumps(details),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  save_json(ROOT/'failure.json',dict(stage='preflight',error=repr(e),traceback=traceback.format_exc()));raise
