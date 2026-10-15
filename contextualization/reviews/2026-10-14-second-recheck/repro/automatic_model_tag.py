"""Actual CPU evaluation proves autodiscovered tag is not used for provenance; local only."""
import contextlib, hashlib, io, json, os, shutil, sys, tempfile
from pathlib import Path
ROOT=Path('/home/danielaush/Documents/peacechat')
sys.path[:0]=[str(ROOT/'contextualization'),str(ROOT)]
os.environ['CTX_TRACKING']='0'
import torch
torch.set_num_threads(1)
import eval_beliefs as E
from nanochat.checkpoint_manager import find_largest_model
E.T.load_env=lambda:[]
src=Path('/tmp/ctx_review4_smoke/build')
root=Path(tempfile.mkdtemp(prefix='ctx_review4_autotag_'))
shutil.copy(src/'build_summary.json',root)
shutil.copytree(src/'probe_sets',root/'probe_sets')
arm=root/'arm_C';shutil.copytree(src/'arm_C/tokenizer',arm/'tokenizer')
d=arm/'base_checkpoints/d2_s7';d.mkdir(parents=True)
for f in ['model_000267.pt','meta_000267.json']:
 shutil.copy(src/'arm_C/base_checkpoints/d2_s7'/f,d/f)
actual_sha=hashlib.sha256((d/'model_000267.pt').read_bytes()).hexdigest()
args=['--exp-dir',str(root),'--arm','C','--limit','2','--batch-size','4','--no-tracking']
for label in ['replicate_only']:
 selected=find_largest_model(arm/'base_checkpoints')
 assert selected=='d2_s7',selected
 with contextlib.redirect_stdout(io.StringIO()) as log:
  csvpath=E.main(args)
 meta=json.loads(Path(csvpath).with_suffix('.meta.json').read_text())
 print(label,json.dumps({'selected_tag':selected,'recorded_tag':meta['model_tag'],'actual_model_sha256':actual_sha,'recorded_model_sha256':meta['checkpoint']['model_sha256'],'recorded_model_path':meta['checkpoint']['model_path'],'hash_matches_scored_model':meta['checkpoint']['model_sha256']==actual_sha,'output':csvpath}))
print('ROOT',root)
