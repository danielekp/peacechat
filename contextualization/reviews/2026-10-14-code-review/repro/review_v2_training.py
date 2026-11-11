import sys, os, json, tempfile, shutil
from pathlib import Path
import numpy as np
sys.path[:0] = ['/home/danielaush/Documents/peacechat/contextualization', '/home/danielaush/Documents/peacechat']
from tests.conftest import small_bpe_tokenizer
from tests.test_prepacked_loader import build_experiment, ARMS, T, load_rows
from ctx.prepack import prepack_experiment, verify_prepack, schedule_exposure
from ctx.exposure import read_exposure_csv, write_exposure_csv, cross_arm_report
from nanochat.dataloader import prepacked_distributed_data_loader_with_state
import exposure_audit
root = Path(tempfile.mkdtemp(prefix='ctx_training_review_'))
tok = small_bpe_tokenizer.__wrapped__(None)
build_experiment(root, tok)
result = prepack_experiment(root, ARMS, tok, T, buffer_size=8, tokenizer_batch_size=6, rows_per_file=16)
print('FIXTURE',root,'n_rows',result['n_rows'])
# incomplete indexes
p=root/'arm_X/prepacked/occurrence_exposure.csv'; original=p.read_text()
rows=read_exposure_csv(p); write_exposure_csv(p,rows[1:])
print('MISSING_OCCURRENCE_VERIFIED',verify_prepack(root,ARMS))
p.write_text(original)
# corrupted masks, manifests still intact
m=root/'arm_R/prepacked/mask_00000.npy'; mask=np.load(m); np.save(m,np.zeros_like(mask))
print('ZEROED_MASK_VERIFIED',verify_prepack(root,ARMS))
x,y,st=next(prepacked_distributed_data_loader_with_state(2,T,'train',device='cpu',prepacked_dir=str(m.parent)))
print('ZEROED_MASK_LOADER_ACTIVE_TARGETS',int((y!=-1).sum()))
np.save(m,mask)
# audit mismatch seed test: same row indexes, different step scheduling
for arm,seed in [('R',1),('X',2),('E',3)]:
 p=root/f'arm_{arm}/prepacked/prepack_manifest.json'; d=json.loads(p.read_text()); d['data_seed']=seed;p.write_text(json.dumps(d))
rc=exposure_audit.main(['--exp-dir',str(root),'--arms','R','X','E','--mode','prepacked','--B','1','--grad-accum','1','--num-iterations',str(result['n_rows']-1)])
rep=json.loads((root/'exposure/cross_arm.json').read_text())
print('MISMATCHED_STEP_EXIT',rc,'row_agree',rep['n_row_agree'],'step_agree',rep['n_step_agree'],'occurrences',rep['n_occurrences'])
# actual batch target counts; all actual scored targets weighted 1/n_valid rather than 1/(B*T)
R=load_rows(root,'R')[2][:,1:].sum(axis=1); X=load_rows(root,'X')[2][:,1:].sum(axis=1)
print('MASK_COUNTS', [{'row':i,'R':int(R[i]),'X':int(X[i]),'R_token_gradient_factor_relative_to_X':float(X[i]/R[i])} for i in range(len(R)) if R[i]!=X[i]])
# dataset replacement while resuming
ld=prepacked_distributed_data_loader_with_state(2,T,'train',device='cpu',prepacked_dir=str(root/'arm_R/prepacked'))
x,y,state=next(ld); saved=x.clone()
p=root/'arm_R/prepacked/rows_00000.npy'; arr=np.load(p); arr[0,1]+=1;np.save(p,arr)
x2,y2,st2=next(prepacked_distributed_data_loader_with_state(2,T,'train',device='cpu',prepacked_dir=str(p.parent),resume_state_dict=state))
print('ALTERED_RESUME_ACCEPTED',not bool((saved==x2).all()),st2)
