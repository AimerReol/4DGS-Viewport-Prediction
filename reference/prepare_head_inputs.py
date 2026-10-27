"""Prepare Actor1 head caches from the confirmed playback CSV; never fit a model."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from motion_query_pose.settings import RECORD_SPLITS,HORIZONS

def prepare(csv,output):
    raw=np.genfromtxt(csv,delimiter=',',skip_header=1)
    assert raw.shape==(35*176,8) and np.isfinite(raw).all()
    np.testing.assert_array_equal(raw[:,0].reshape(35,176),np.tile(np.arange(1,177),(35,1)))
    poses=raw[:,1:].reshape(35,176,7).copy()
    norm=np.linalg.norm(poses[...,3:],axis=-1,keepdims=True)
    assert np.all(norm>1e-12);poses[...,3:]/=norm
    starts=np.arange(49,126);users=np.repeat(np.arange(35),len(starts));origins=np.tile(starts,35)
    target=origins[:,None]+HORIZONS;x=[];y=[]
    for u,t,frames in zip(users,origins,target):
        past=poses[u,t-49:t+1];r=Rotation.from_quat(past[:,3:]).as_matrix();cur=r[-1]
        x.append(np.concatenate([(past[:,:3]-past[-1,:3])@cur,np.einsum('ij,tjk->tik',cur.T,r)[:,:,:2].reshape(50,6)],-1))
        future=poses[u,frames];q=Rotation.from_quat(past[-1,3:])
        y.append(np.concatenate([(future[:,:3]-past[-1,:3])@cur,(q.inv()*Rotation.from_quat(future[:,3:])).as_rotvec()],-1))
    x=np.asarray(x,np.float32);y=np.asarray(y,np.float32)
    output.mkdir(parents=True,exist_ok=True)
    destinations=[output/'poses.npz']+[output/f'{s}_inputs.npz' for s in RECORD_SPLITS]
    if any(p.exists() for p in destinations):raise FileExistsError('Use a fresh output directory')
    np.savez_compressed(output/'poses.npz',poses=poses)
    for split,records in RECORD_SPLITS.items():
        ids={'split_'+k:np.flatnonzero(np.isin(users,np.asarray(v)-1)) for k,v in records.items()}
        tr=ids['split_train'];xm=x[tr].mean((0,1));xs=x[tr].std((0,1)).clip(.001)
        np.savez_compressed(output/f'{split}_inputs.npz',x=((x-xm)/xs).astype(np.float32),y=y,
            users=users,origins=origins,target=target,x_mean=xm,x_std=xs,
            y_scale=y[tr].std((0,1)).clip(.001),
            loss_scales=np.array([(y[tr,:,:3]**2).sum(-1).mean(),(y[tr,:,3:]**2).sum(-1).mean()]).clip(1e-6),**ids)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.csv,a.output)
