import json,time
import numpy as np
import torch
from .experiment import *
from .rendering import NativeRenderer,Supervision,probes
def main():
 runtime();wait_for_resources('training_only_render_scale');r=NativeRenderer();all_scales={}
 assert json.loads((ROOT/'ours_preflight.json').read_text())['complete']
 for split in SPLITS:
  a=dict(np.load(DATA/'cache'/f'{split}_inputs.npz'));sup=Supervision(r,a);chosen=[]
  for u in np.unique(a['users'][a['split_train']]):
   ids=a['split_train'][a['users'][a['split_train']]==u]
   chosen.extend(ids[np.rint(np.linspace(0,len(ids)-1,RENDER['calibration_windows_per_record'])).astype(int)].tolist())
  assert set(chosen)<=set(a['split_train']);rng=np.random.default_rng(2026101221);values=[];began=time.time()
  with torch.no_grad():
   for j,i in enumerate(chosen):
    row=[]
    for h in range(len(HORIZONS)):
     row.append(float(r.pair(int(a['target'][i,h]),sup.current_position[i],sup.current_rotation[i],torch.zeros(6,device='cuda'),torch.tensor(a['y'][i,h],device='cuda'),probes(rng),'RenderFull')))
    values.append(row)
    if j%12==0:status('calibrating_render_scale',split=split,windows_completed=j+1,total_windows=len(chosen))
  den=np.maximum(np.mean(values,axis=0),1e-6)
  all_scales[split]=dict(denominator=den.tolist(),window_indices=chosen,recordings=(a['users'][chosen]+1).tolist(),per_window_discrepancy=values,seconds=time.time()-began)
  print('CALIBRATED',split,den.tolist(),flush=True)
 save_json(ROOT/'render_scale.json',dict(complete=True,data_manifest_sha256=sha(DATA/'data_manifest.json'),reference='Hold observed origin pose versus measured future pose; full Gaussian renderer; training records only; same denominators for all variants.',splits=all_scales));status('render_scale_ready')
if __name__=='__main__':main()
