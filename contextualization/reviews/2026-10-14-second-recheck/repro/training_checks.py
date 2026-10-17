import os,sys,json
from pathlib import Path
import numpy as np
sys.path[:0]=['/home/danielaush/Documents/peacechat/contextualization','/home/danielaush/Documents/peacechat']
from nanochat.dataloader import prepacked_distributed_data_loader_with_state
from nanochat.tokenizer import RustBPETokenizer
from ctx.prepack import loss_weighting_report
from tests.test_seeds import base_train
root=Path('/tmp/ctx_review3_resume_zhh6najx')
assert root.exists()
tok=RustBPETokenizer.from_directory(str(root/'arm_R/tokenizer'))
B,W,GA,N,seed=3,2,4,5,13
horizon=GA*N
lw=loss_weighting_report(root,['R','X'],B=B,grad_accum_steps=GA,num_iterations=N,world_size=W,data_seed=seed)
checks=[]
for arm in ['R','X']:
 for rank in range(W):
  os.environ.update(RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE=str(W))
  make=lambda state=None:prepacked_distributed_data_loader_with_state(B,96,'train',device='cpu',data_seed=seed,prepacked_dir=str(root/f'arm_{arm}/prepacked'),tokenizer=tok,resume_state_dict=state)
  ld=make(); states=[]; batches=[]; counts=[]
  for m in range(horizon+1):
   x,y,st=next(ld); states.append(st.copy()); batches.append((x.clone(),y.clone())); counts.append(int((y!=-1).sum()))
   # 20 stored rows, B*W=6:18 trained rows/epoch,2 dropped.
   assert st['epoch']*18+st['cursor']==m*B*W,(arm,rank,m,st)
  for m in [1,2,3,4,7,18,19]:
   resumed=make(states[m]);x,y,st=next(resumed)
   assert bool((x==batches[m][0]).all()) and bool((y==batches[m][1]).all()) and st==states[m]
  actual=np.array(counts[:horizon]); reported=lw['per_arm'][arm]['per_rank'][str(rank)]
  assert reported=={'n':horizon,'min':float(actual.min()),'mean':float(actual.mean()),'max':float(actual.max()),'total':int(actual.sum())},(reported,actual.tolist())
  checks.append({'arm':arm,'rank':rank,'n_microbatches':horizon,'resume_positions_tested':7,'true_rank_count_min':int(actual.min()),'true_rank_count_max':int(actual.max()),'report_matches_actual_loader':True})
for key in ['RANK','LOCAL_RANK','WORLD_SIZE']:os.environ.pop(key,None)
r=base_train(root/'arm_R','--num-iterations','10','--save-every','5','--init-seed','7','--model-tag','d2_ga','--resume-from-step','5','--total-batch-size','384',expect=1)
assert "the requested training schedule differs from the checkpoint's" in r.stderr,r.stderr
print(json.dumps({'loader_cursor_ddp_rank_and_epoch_tail_checks':checks,'old_changed_accumulation_reproduction_now_rejected':True,'rejection':r.stderr.splitlines()[-1]},indent=2))
