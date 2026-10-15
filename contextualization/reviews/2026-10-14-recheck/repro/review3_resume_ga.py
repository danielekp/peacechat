import os,sys,json,tempfile
from pathlib import Path
sys.path[:0]=['/home/danielaush/Documents/peacechat/contextualization','/home/danielaush/Documents/peacechat']
from tests.conftest import small_bpe_tokenizer
from tests.test_prepacked_loader import ARMS,build_experiment,prepack,T
from tests.test_seeds import _write_token_bytes,base_train,meta
from exposure_audit import prepacked_preflight
from nanochat.dataloader import prepacked_distributed_data_loader_with_state
root=Path(tempfile.mkdtemp(prefix='ctx_review3_resume_'))
tok=small_bpe_tokenizer.__wrapped__(None)
build_experiment(root,tok)
for a in ARMS:
 tok.save(str(root/f'arm_{a}/tokenizer'));_write_token_bytes(tok,root/f'arm_{a}/tokenizer')
packed=prepack(root,tok);N=packed['n_rows']//2
prepacked_preflight(root,ARMS,B=2,grad_accum_steps=1,num_iterations=N,world_size=1,data_seed=0,out_dir=root/'exposure',check_hashes=True)
r=base_train(root/'arm_R','--num-iterations',str(N),'--save-every','5','--init-seed','7','--model-tag','d2_ga')
print('FIRST_TRAIN_OK',N,flush=True)
prepacked_preflight(root,ARMS,B=2,grad_accum_steps=2,num_iterations=N,world_size=1,data_seed=0,out_dir=root/'exposure',check_hashes=True)
r=base_train(root/'arm_R','--num-iterations',str(N),'--save-every','5','--init-seed','7','--model-tag','d2_ga','--resume-from-step','5','--total-batch-size','384')
actual=meta(root/'arm_R','d2_ga',N)
ld=prepacked_distributed_data_loader_with_state(2,T,'train',device='cpu',data_seed=0,prepacked_dir=str(root/'arm_R/prepacked'),tokenizer=tok)
for i in range(N*2+1):
 x,y,expected=next(ld)
print('RESUME_GA_REPRO',json.dumps({'root':str(root),'preflight':actual['preflight'],'step':actual['step'],'new_total_batch_size':actual['total_batch_size'],'actual_state':actual['dataloader_state_dict'],'expected_state_from_passed_preflight':expected,'states_match':actual['dataloader_state_dict']==expected}),flush=True)
