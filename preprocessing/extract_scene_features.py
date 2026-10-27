import fcntl,json,math,sys,time,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

PARENT=Path(os.environ['MQP_ACTOR1_PARENT'])
SOURCE=Path(os.environ['MQP_SCENE_DIR'])
ROOT=Path(os.environ['MQP_DATA_DIR'])
PROJECT=Path(os.environ['MQP_BACKEND_DIR'])
sys.path.insert(0,str(PROJECT))
from aps import backend as be
from aps.common import atomic_json as save,file_hash as sha,free_check
K=64;T=176;FPS=10;H=np.array([2,5,10,20,30,50])
def status(stage,**kw):
    d=dict(stage=stage,pid=__import__('os').getpid(),updated_at=time.time(),**kw);save(ROOT/'preparation_status.json',d);print(json.dumps(d),flush=True)
def quat(q):
    q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True).clamp_min(1e-12);w,x,y,z=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1).reshape(-1,3,3)
def pack(c):return c[:,[0,0,0,1,1,2],[0,1,2,1,2,2]]
def pool(x,labels):
    n=np.bincount(labels,minlength=K);s=np.zeros((K,x.shape[1]));q=np.zeros_like(s)
    np.add.at(s,labels,x);np.add.at(q,labels,x*x);mean=s/np.maximum(n[:,None],1)
    return mean,np.sqrt(np.maximum(q/np.maximum(n[:,None],1)-mean*mean,0)),n
def grouping(x):
    first=np.argmin(np.linalg.norm(x-np.median(x,axis=0),axis=1));centers=[x[first]];distance=np.full(len(x),np.inf)
    for _ in range(1,K):distance=np.minimum(distance,((x-centers[-1])**2).sum(1));centers.append(x[np.argmax(distance)])
    centers=np.asarray(centers)
    for _ in range(60):
        labels=cdist(x,centers,'sqeuclidean').argmin(1)
        new=np.array([x[labels==i].mean(0) if (labels==i).any() else centers[i] for i in range(K)])
        if np.max(abs(new-centers))<1e-6:break
        centers=new
    labels=cdist(x,centers,'sqeuclidean').argmin(1);assert len(np.unique(labels))==K
    return centers,labels
def main():
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'preparation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'data_checks.json').exists():assert json.loads((ROOT/'data_checks.json').read_text())['complete'];return
    for folder in ['cache','grouping','coverage_images','weights','curves','predictions','scores']:(ROOT/folder).mkdir(exist_ok=True)
    import subprocess
    live=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip();assert not live,live
    done=json.loads((SOURCE/'done.json').read_text());assert done['complete'] and done['frames']==T
    checkpoint=SOURCE/'base/model.pt';assert sha(checkpoint)==done['model_sha256']
    protocol=json.loads((PARENT/'head_protocol.json').read_text());cal=json.loads((PARENT/'accepted_alignment_A.json').read_text());assert cal['accepted']
    timeline=json.loads((SOURCE/'timeline.json').read_text())['rows'];assert len(timeline)==T
    source_time=np.array([x['reconstruction_time'] for x in timeline]);assert np.all((source_time>=0)&(source_time<=1))
    protected=json.loads((SOURCE/'protected_sha256.json').read_text())
    for n,h in protected.items():assert sha(n)==h,n
    cfg=json.loads((SOURCE/'config.json').read_text());torch.set_num_threads(4);be.activate()
    model,_=be.restore_model(cfg,be.load_torch(checkpoint)['model'])
    for p in be.all_params(model):p.requires_grad_(False)
    canonical_all=model._xyz.detach().cpu().numpy();valid=np.flatnonzero((model._opacity.sigmoid().detach().cpu().numpy().ravel()>=.005)&np.isfinite(canonical_all).all(1))
    assert len(valid)>=8192
    ids=np.sort(np.random.default_rng(20260929).choice(valid,8192,replace=False));canonical=canonical_all[ids].astype(float);sites,members=grouping(canonical)
    labels=cKDTree(sites).query(canonical_all,workers=2)[1];np.testing.assert_array_equal(labels[ids],members)
    np.savez_compressed(ROOT/'grouping/grouping.npz',canonical=canonical,sites=sites,gaussian_ids=ids,anchor_members=np.broadcast_to(members,(T,len(ids))))
    all_head={s:dict(np.load(PARENT/'head_cache'/f'{s}_inputs.npz')) for s in ['R1','R2','R3']}
    head=all_head['R1'];poses=np.load(PARENT/'head_cache/poses.npz')['poses'];assert poses.shape==(35,T,7)
    frames=np.unique(head['target']);needed=len(frames)*len(labels)*11*4;free_check(ROOT,5,needed)
    fieldfile=ROOT/'cache/render_fields.npy';tmp=fieldfile.with_suffix('.partial.npy')
    assert not fieldfile.exists(),'Inspect existing fields before resuming preparation'
    field=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(frames),len(labels),11));frameidx={int(f):i for i,f in enumerate(frames)}
    np.savez_compressed(ROOT/'cache/render_index.npz',labels=labels,frames=frames,fake_depth=np.random.default_rng(2026101220).uniform(.5,1.5,len(labels)).astype(np.float32))
    rot=Rotation.from_quat(poses[:,:,3:].reshape(-1,4)).as_matrix().reshape(35,T,3,3)
    A=np.asarray(cal['matrix']);b=np.asarray(cal['translation']);flip=np.diag([1,-1,1]);tanfov=np.tan(np.deg2rad(cal['fov_degrees']/2))
    def camera(u,f,side):
        w=flip@rot[u,f].T@A;center=A.T@(poses[u,f,:3]-b);focal=side/(2*tanfov)
        return SimpleNamespace(R=w.astype(np.float32),t=(-w@center).astype(np.float32),c=center.astype(np.float32),K=np.array([[focal,0,(side-1)/2],[0,focal,(side-1)/2],[0,0,1]],np.float32),width=side,height=side)
    arrays={k:[] for k in ['xyz','rgb','cov','opacity','scale']};coverage=[];started=time.time()
    for f,t in enumerate(source_time):
        with torch.no_grad():
            raw=be.deformed(model,float(t));x,s,q,o,sh=raw
            assert all(len(v)==len(labels) and torch.isfinite(v).all() for v in raw)
            R=quat(q);cov=R@torch.diag_embed((2*s).exp())@R.transpose(-1,-2);packed=pack(cov)
            vals=[x,torch.clamp(.28209479177387814*sh[:,0]+.5,0,1),packed,o.sigmoid(),s.exp()]
            for k,v in zip(arrays,vals):arrays[k].append(v[ids].cpu().numpy())
            if f in frameidx:field[frameidx[f]]=torch.cat([x,packed,o.sigmoid(),(2*s.mean(-1,keepdim=True)).exp()],-1).cpu().numpy()
            for u in range(35):
                image=be.packed_render(raw,be.camera_view(camera(u,f,128),float(t)),degree=model.active_sh_degree)['render']
                assert torch.isfinite(image).all()
                rgb=np.round(image.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                coverage.append(dict(record=u+1,csv_frame=f+1,source_frame=timeline[f]['inferred_source_frame_0based'],foreground_fraction=float((rgb.min(-1)<240).mean())))
                if f in [0,49,88,125,175]:
                    image=be.packed_render(raw,be.camera_view(camera(u,f,420),float(t)),degree=model.active_sh_degree)['render']
                    Image.fromarray(np.round(image.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save(ROOT/'coverage_images'/f'record{u+1:02d}_csv{f+1:03d}.png')
        if f%8==0 or f==175:status('extracting_reconstructed_scene',frames_done=f+1,frames_total=T,coverage_renders=len(coverage),seconds=time.time()-started)
    field.flush();del field;tmp.replace(fieldfile)
    arrays={k:np.asarray(v) for k,v in arrays.items()};xyz=arrays['xyz'].astype(float)
    np.savez_compressed(ROOT/'extracted.npz',**arrays,canonical=canonical,gaussian_ids=ids,poses=poses,reconstruction_times=source_time)
    velocity=np.concatenate([np.zeros_like(xyz[:1]),np.diff(xyz,axis=0)]);acceleration=np.concatenate([np.zeros_like(xyz[:1]),np.diff(velocity,axis=0)])
    attr=np.concatenate([xyz,arrays['rgb'],arrays['cov'],arrays['opacity'],arrays['scale'],velocity,acceleration,np.linalg.norm(velocity,axis=-1,keepdims=True),xyz-canonical[None]],-1)
    raw_features=[];centers=[];radii=[];counts=[];edges=[]
    for f in range(T):
        mean,std,n=pool(attr[f],members);raw_features.append(np.concatenate([mean[:,:3],std[:,:3],mean[:,3:6],std[:,3:6],mean[:,6:],std[:,6:],np.log1p(n[:,None])],-1))
        centers.append(mean[:,:3]);radii.append(np.linalg.norm(std[:,:3],axis=-1));counts.append(n)
        dist=cdist(mean[:,:3],mean[:,:3]);np.fill_diagonal(dist,np.inf);edges.append(np.argsort(dist,axis=1)[:,:6])
    raw_features=np.array(raw_features);centers=np.array(centers);radii=np.array(radii);counts=np.array(counts)
    canonical_center=pool(canonical,members)[0];previous=np.maximum(np.arange(T)-5,0);disp=centers-centers[previous]
    primitive=np.concatenate([centers,np.broadcast_to(canonical_center,centers.shape),disp/np.maximum(np.arange(T)-previous,1)[:,None,None],disp],-1)
    origins_one=np.arange(49,126);starts=np.repeat(origins_one,len(H));stops=(origins_one[:,None]+H).ravel()
    endpoints=stops[:,None]-np.arange(5,-1,-1);intervals=starts[:,None]+np.rint((stops-starts)[:,None]*np.linspace(0,1,6)).astype(int)
    used=np.unique(np.concatenate([endpoints.ravel(),intervals.ravel()]));fm,fs=raw_features[used].mean((0,1)),raw_features[used].std((0,1)).clip(1e-5);nm,ns=primitive[used].mean((0,1)),primitive[used].std((0,1)).clip(1e-5)
    features=np.concatenate([np.clip((raw_features-fm)/fs,-8,8),np.clip((primitive-nm)/ns,-8,8)],-1).astype(np.float32);assert features.shape==(T,64,65)
    np.savez_compressed(ROOT/'cache/scene.npz',features=features,edges=np.array(edges),valid=counts>0,feature_mean=fm,feature_std=fs,node_mean=nm,node_std=ns,normalizer_frames=used,endpoint_frames=endpoints,endpoint_times=((endpoints-starts[:,None])/FPS).astype(np.float32))
    users=head['users'];origins=head['origins'];target=head['target'];np.testing.assert_array_equal(target,origins[:,None]+H)
    contexts=np.tile(np.arange(len(origins_one)*len(H)).reshape(len(origins_one),-1),(35,1));world=centers@A.T+b;geometry=[]
    for u,o,ff in zip(users,origins,target):
        cur=rot[u,o];rel=(world[ff]-poses[u,o,:3])@cur;distance=np.linalg.norm(rel,axis=-1,keepdims=True)
        geometry.append(np.concatenate([rel,rel/np.maximum(distance,.001),distance,radii[ff,:,None],np.log1p(counts[ff,:,None]),np.broadcast_to((H/FPS)[:,None,None],(len(H),64,1))],-1))
    geometry=np.array(geometry,np.float32);slots=np.arange(50).reshape(5,10)[:,np.rint(np.linspace(0,9,6)).astype(int)].ravel();past_frames=origins[:,None]-49+slots
    observed=poses[users[:,None],past_frames];rotations=rot[users[:,None],past_frames]
    rel=np.einsum('btki,btij->btkj',world[past_frames]-observed[:,:,None,:3],rotations);distance=np.linalg.norm(rel,axis=-1,keepdims=True);past=np.concatenate([rel,rel/np.maximum(distance,.001),distance],-1)
    for split,a in all_head.items():
        for k in ['users','origins','target']:np.testing.assert_array_equal(a[k],head[k])
        tr=a['split_train'];gm,gs=geometry[tr].mean((0,1,2)),geometry[tr].std((0,1,2)).clip(.001)
        a.update(geometry=np.clip((geometry-gm)/gs,-8,8).astype(np.float32),past_geometry=np.clip((past-gm[:7])/gs[:7],-8,8).astype(np.float32),geometry_mean=gm,geometry_std=gs,contexts=contexts)
        assert a['x'].shape==(2695,50,9) and a['y'].shape==(2695,6,6)
        assert all(np.isfinite(v).all() for v in a.values())
        np.savez_compressed(ROOT/'cache'/f'{split}_inputs.npz',**a)
    np.savez_compressed(ROOT/'cache/poses.npz',poses=poses)
    alignment=dict(cal,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
    save(ROOT/'data_manifest.json',dict(complete=True,scene='Actor1 matched reconstruction',frames=T,fps=FPS,history_frames=50,horizons=H.tolist(),origins_per_record=77,records=35,alignment=alignment,
        source_reconstruction=str(SOURCE),timeline_sha256=sha(SOURCE/'timeline.json'),reconstruction_times=source_time.tolist(),
        input_contract='Observed head history ends at origin. Known video through each target is available. Measured future heads occur only in targets/supervision/evaluation.',
        motion_features='Backward differences per native playback step; primitive lag 5 playback samples (0.5 s). Six endpoint samples separated by 0.1 s; time encodings in seconds.',
        gaussian_ids_sha256=sha(ROOT/'grouping/grouping.npz'),cache_sha256={p.name:sha(p) for p in (ROOT/'cache').glob('*.npz')},
        scope='Same accepted estimated calibration A for all methods; fixed source time mapping; original image and calibration limitations retained.'))
    save(ROOT/'field_manifest.json',dict(complete=True,gaussians=len(labels),frames=frames.tolist(),checkpoint_sha256=sha(checkpoint),fields_sha256=sha(fieldfile),index_sha256=sha(ROOT/'cache/render_index.npz'),fixed_labels_across_frames=True))
    save(ROOT/'coverage.json',dict(complete=True,rows=coverage,views=6160,fixed_images=175,
        zero_foreground_by_record={str(u):sum(r['record']==u and r['foreground_fraction']==0 for r in coverage) for u in range(1,36)},all_finite=True,gaussians=len(labels),same_canonical_rows_at_all_times=True))
    for n,h in protected.items():assert sha(n)==h,n
    save(ROOT/'data_checks.json',dict(complete=True,all_frames=176,all_head_cache_fields_preserved=True,canonical_identity_from_one_checkpoint=True,shared_labels_all_frames=True,
        source_model_sha256=sha(checkpoint),accepted_alignment_sha256=sha(PARENT/'accepted_alignment_A.json'),head_protocol_sha256=sha(PARENT/'head_protocol.json'),
        all_protected_files_unchanged=True,no_test_scores_used=True,seconds=time.time()-started,script_sha256=sha(__file__)))
    status('scene_inputs_ready',frames=176,records=35,training_started=False)
if __name__=='__main__':main()
