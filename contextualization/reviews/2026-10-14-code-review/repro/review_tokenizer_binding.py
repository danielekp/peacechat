import sys,shutil,json,tempfile
from pathlib import Path
sys.path[:0]=['/home/danielaush/Documents/peacechat/contextualization','/home/danielaush/Documents/peacechat']
from tests.conftest import small_bpe_tokenizer,HISTORICAL_TOKENIZER_DIR
from tests.test_prepacked_loader import build_experiment,ARMS,T
from ctx.protocol import sha256_file
from ctx.prepack import prepack_experiment
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataloader import prepacked_distributed_data_loader_with_state
root=Path(tempfile.mkdtemp(prefix='ctx_tok_bind_review_'))
small=small_bpe_tokenizer.__wrapped__(None)
override=root/'override_tokenizer'; small.save(str(override))
base=HISTORICAL_TOKENIZER_DIR/'tokenizer/tokenizer.pkl'
assert base.exists(),base
for a in ARMS:
 t=root/f'arm_{a}/tokenizer';t.mkdir(parents=True);shutil.copy2(base,t/'tokenizer.pkl')
build_experiment(root,small)
r=prepack_experiment(root,ARMS,small,T,buffer_size=8,tokenizer_batch_size=6,rows_per_file=16)
actual=sha256_file(override/'tokenizer.pkl'); recorded=r['arms']['R']['tokenizer_sha256']; declared=sha256_file(base)
print('TOKENIZER_BINDING_REPRO',json.dumps({'root':str(root),'actual_used_sha256':actual,'declared_arm_sha256':declared,'prepack_manifest_sha256':recorded,'matches_actual':actual==recorded,'matches_unrelated_arm_file':declared==recorded,'verify_problems':r['verify']['n_problems']}))
x,y,s=next(prepacked_distributed_data_loader_with_state(1,T,'train',device='cpu',prepacked_dir=str(root/'arm_R/prepacked')))
print('WRONG_TOKENIZER_ROWS_LOADED',tuple(x.shape))
