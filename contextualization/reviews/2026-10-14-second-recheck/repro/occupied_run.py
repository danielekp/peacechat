"""Repeat the same launcher command with fake training and real dry-run evaluation; offline."""
import contextlib,io,json,sys,tempfile
from pathlib import Path
sys.path[:0]=[str(Path.cwd()/'contextualization'),str(Path.cwd())]
import run_arm as R
import eval_beliefs as E
from ctx import tracking as T
from ctx.factpool_v2 import build_semantic_pool,split_groups,assign_injection,to_probe_records
from ctx.templates import RenderPolicy
from ctx.protocol import PROTOCOL_V2
root=Path(tempfile.mkdtemp(prefix='ctx_review4_retry_'));(root/'arm_C').mkdir();p=root/'probe_sets';p.mkdir()
g=build_semantic_pool(30,1234);tr,he=split_groups(g,.3,1234);assign_injection(tr,[1],1234);inj,held=to_probe_records(tr,he,['C','R'])
for name,records in [('injected_facts',inj),('heldout_facts',held)]:
 (p/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
(root/'build_summary.json').write_text(json.dumps({'protocol':PROTOCOL_V2,'config':{'seed':1234,'render_policy':RenderPolicy.fixed_rendering().to_dict()}}))
T._ENV_LOADED=True;T.load_env=lambda:[];T.tracking_enabled=lambda:False
T.wandb_enabled=lambda:True;T.hf_enabled=lambda:False;T.wandb_project=lambda:'fake'
T.wandb_run_url=lambda run_id:'https://example.invalid/'+run_id
runids=iter(['run7','run7retry']);T.new_run_id=lambda:next(runids)
T.export_manifest=lambda *a,**kw:root/'fake_export.json'
seen=[]
def sh(cmd,**kw):
 cmd=list(map(str,cmd))
 if 'scripts.base_train' in cmd:
  tag=next(a.split('=',1)[1] for a in cmd if a.startswith('--model-tag='))
  seed=int(next(a.split('=',1)[1] for a in cmd if a.startswith('--init-seed=')))
  ck=root/'arm_C/base_checkpoints'/tag;ck.mkdir(parents=True,exist_ok=True)
  (ck/'model_000010.pt').write_text(kw['env']['WANDB_RUN_ID'])
  (ck/'meta_000010.json').write_text(json.dumps({'init_seed':seed,'init_param_sha256':str(seed)*64,'step':10,'wandb_run':{'id':kw['env']['WANDB_RUN_ID']}}))
 else:
  out=E.main(cmd[2:]+['--dry-run','--no-tracking'])
  seen.append({'model_tag':cmd[cmd.index('--model-tag')+1],'csv':str(out),'wandb_run_id':T.read_run_info(root,'C','d2_s7').get('wandb_run_id')})
R.sh=sh
failures=[]
for seed in [7,7]:
 sys.argv=['run_arm.py','--arm','C','--exp-dir',str(root),'--depth','2','--init-seed',str(seed),'--data-loader','bestfit','--skip-upload']
 try:
  with contextlib.redirect_stdout(io.StringIO()):R.main()
 except SystemExit as e:failures.append({'seed':seed,'error':str(e)})
print('LAUNCH_RESULTS',json.dumps(seen))
print('RETRY_FAILURE',json.dumps(failures))

print('FINAL_RUN_INFO_ID',T.read_run_info(root,'C','d2_s7')['wandb_run_id'])
print('CHECKPOINT_BYTES', (root/'arm_C/base_checkpoints/d2_s7/model_000010.pt').read_text())
print('EVALUATION_RUN_ID',json.loads((root/'eval/seed_7/beliefs_armC.meta.json').read_text())['run']['wandb_run_id'])
print('ROOT',root)
