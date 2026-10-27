"""Decode the six relative pose heads to absolute world-space poses."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .models import Predictor
from .experiment import tensors,encode,forward
from .settings import DATA,ROOT,HORIZONS,CONFIG
from .pose_metrics import pose_errors
from .data import sha,save_json

@torch.no_grad()
def predict_checkpoint(checkpoint,split,part='test',device='cuda'):
    # Load only your own trusted checkpoints: the frozen format contains NumPy objects.
    saved=torch.load(checkpoint,map_location=device,weights_only=False)
    metadata=saved.get('metadata',{})
    if metadata.get('split')!=split:raise ValueError('Checkpoint split does not match --split')
    if metadata.get('data_manifest_sha256')!=sha(DATA/'data_manifest.json'):raise ValueError('Checkpoint data manifest mismatch')
    model=Predictor(metadata.get('method','NoRender')).to(device).eval();model.load_state_dict(saved['state'],strict=True)
    d=tensors(split,device);a=d['arrays'];ids=a['split_'+part];encoded=encode(model,d);chunks=[]
    for begin in range(0,len(ids),CONFIG['batch']):chunks.append(forward(model,d,ids[begin:begin+CONFIG['batch']],encoded)['pose'])
    relative=(torch.cat(chunks)*d['y_scale']).cpu().numpy()
    poses=np.load(DATA/'cache/poses.npz')['poses'];users=a['users'][ids];origins=a['origins'][ids]
    current=poses[users,origins];rotation=Rotation.from_quat(current[:,3:])
    position=current[:,None,:3]+np.einsum('bij,bhj->bhi',rotation.as_matrix(),relative[...,:3])
    quaternion=(Rotation.from_quat(np.repeat(current[:,3:],len(HORIZONS),axis=0))*Rotation.from_rotvec(relative[...,3:].reshape(-1,3))).as_quat().reshape(-1,len(HORIZONS),4)
    truth=poses[users[:,None],a['target'][ids]]
    metrics=pose_errors(position,quaternion,truth[...,:3],truth[...,3:])
    output=ROOT/'predictions'/f'{checkpoint.stem}_{part}.npz'
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
    np.savez_compressed(output,predicted_position=position,predicted_quaternion_xyzw=quaternion,
        relative_pose=relative,target_relative_pose=a['y'][ids],target_position=truth[...,:3],target_quaternion_xyzw=truth[...,3:],
        recordings=users+1,origins=origins,horizons=HORIZONS,history_frames=50,split=split)
    report={'checkpoint_sha256':sha(checkpoint),'prediction_sha256':sha(output),'windows':len(ids),'metrics':[
        {'horizon_seconds':float(h/10),**{k:float(v[:,j].mean()) for k,v in metrics.items()}} for j,h in enumerate(HORIZONS)]}
    save_json(output.with_suffix('.json'),report);return report
