from pathlib import Path
import numpy as np
from .data import sha,save_json,HORIZONS
import os
DATA=Path(os.environ.get('MQP_DATA_DIR','data/prepared')).resolve()
ROOT=Path(os.environ.get('MQP_OUTPUT_DIR','runs/current')).resolve()
PARENT=DATA
OLD=DATA
SOURCE=Path(os.environ.get('MQP_SCENE_DIR','data/reconstruction')).resolve()
PROJECT=Path(os.environ.get('MQP_BACKEND_DIR','external/renderer-project')).resolve()
METHODS=['Baseline','RenderFull']
SPLITS=['R1','R2','R3'];SEEDS=[0,1,2];TOTAL=18
CONFIG=dict(epochs=80,patience=12,batch=96,lr=.001,weight_decay=1e-4,grad_clip=1,device='cuda')
RENDER=dict(side=48,fov_degrees=50.,samples_per_batch=6,channels=3,lambda_max=.1,warmup_epochs=5,calibration_windows_per_record=3,eval_origins_per_record=6,eval_horizons=[10,30,50])
SPLIT_SEED=20261007
groups=np.random.default_rng(SPLIT_SEED).permutation(np.arange(1,36)).reshape(5,7).tolist()
RECORD_SPLITS={name:dict(test=sorted(groups[i]),val=sorted(groups[(i+1)%5]),train=sorted(sum([group for j,group in enumerate(groups) if j not in [i,(i+1)%5]],[]))) for i,name in enumerate(SPLITS)}
ALIASES={'Baseline':'NoRender','RenderFull':'Full'}
def name(method,split,seed):return f'{split}_{ALIASES.get(method,method)}_H50_seed{seed}'

FIELD_PATH=DATA/'cache/render_fields.npy'
