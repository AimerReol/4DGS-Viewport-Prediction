"""Fail early on incompatible caches; indices refer to playback, not source-video order."""
import json
import numpy as np
from .data import sha,HORIZONS

def inspect_cache(root,splits):
    manifest=json.loads((root/'data_manifest.json').read_text(encoding='utf-8'))
    assert manifest.get('complete') and manifest['fps']==10 and manifest['history_frames']==50
    np.testing.assert_array_equal(manifest['horizons'],HORIZONS)
    with np.load(root/'cache/scene.npz',allow_pickle=False) as s:
        f=s['features'];assert f.ndim==3 and f.shape[1:]==(64,65)
        assert np.isfinite(f).all() and s['edges'].shape==(len(f),64,6)
        assert s['valid'].shape==(len(f),64) and s['valid'].dtype==bool
        assert s['edges'].min()>=0 and s['edges'].max()<64
        ef=s['endpoint_frames'];et=s['endpoint_times'];assert ef.shape==et.shape and ef.shape[1]==6
        assert ef.min()>=0 and ef.max()<len(f) and np.isfinite(et).all()
        frames=len(f);contexts=len(ef)
    with np.load(root/'cache/poses.npz',allow_pickle=False) as p:
        poses=p['poses'];assert poses.ndim==3 and poses.shape[1:]==(frames,7) and np.isfinite(poses).all()
        assert np.all(np.linalg.norm(poses[...,3:],axis=-1)>1e-8)
    result={'frames':frames,'records':len(poses),'data_manifest_sha256':sha(root/'data_manifest.json'),'splits':{}}
    expected_hashes=manifest.get('cache_sha256',{})
    for file in ['scene.npz','poses.npz']+[s+'_inputs.npz' for s in splits]:
        if file in expected_hashes:assert sha(root/'cache'/file)==expected_hashes[file],file+' hash mismatch'
    for split in splits:
        with np.load(root/'cache'/f'{split}_inputs.npz',allow_pickle=False) as a:
            n=len(a['x']);assert a['x'].shape==(n,50,9) and a['y'].shape==(n,6,6)
            assert a['geometry'].shape==(n,6,64,10) and a['past_geometry'].shape==(n,30,64,7)
            np.testing.assert_array_equal(a['target'],a['origins'][:,None]+HORIZONS)
            assert a['origins'].min()>=49 and a['target'].max()<frames
            assert a['users'].min()>=0 and a['users'].max()<len(poses)
            assert a['contexts'].shape==(n,6) and a['contexts'].min()>=0 and a['contexts'].max()<contexts
            np.testing.assert_array_equal(ef[a['contexts']][:,:,-1],a['target'])
            expected_time=(ef[a['contexts']]-a['origins'][:,None,None])/10
            np.testing.assert_allclose(et[a['contexts']],expected_time,rtol=1e-6,atol=1e-6)
            assert all(np.isfinite(a[k]).all() for k in a.files)
            assert a['y_scale'].shape==(6,) and np.all(a['y_scale']>0) and np.all(a['loss_scales']>0)
            sets=[set(a['split_'+p].tolist()) for p in ['train','val','test']]
            assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            assert set.union(*sets)==set(range(n))
            records=[set(a['users'][a['split_'+p]].tolist()) for p in ['train','val','test']]
            assert not(records[0]&records[1] or records[0]&records[2] or records[1]&records[2])
            result['splits'][split]={'windows':n,'part_windows':dict(zip(['train','val','test'],map(len,sets))),'records':dict(zip(['train','val','test'],[sorted(x+1 for x in r) for r in records]))}
    return result
