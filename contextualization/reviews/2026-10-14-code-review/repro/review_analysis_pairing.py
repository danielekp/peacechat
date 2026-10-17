import sys,csv,json,tempfile,contextlib,io
from pathlib import Path
sys.path[:0]=['/home/danielaush/Documents/peacechat/contextualization','/home/danielaush/Documents/peacechat']
import analyze_beliefs as AB
from ctx.factpool_v2 import build_semantic_pool,split_groups,assign_injection,to_probe_records
from ctx.templates import RenderPolicy
from ctx.protocol import PROTOCOL_V2
root=Path(tempfile.mkdtemp(prefix='ctx-review-pairing-'));pdir=root/'probe_sets';edir=root/'eval';pdir.mkdir();edir.mkdir()
g=build_semantic_pool(30,1234);tr,he=split_groups(g,.3,1234);assign_injection(tr,[1],1234);inj,held=to_probe_records(tr,he,['C','R'])
for name,rows in [('injected_facts',inj),('heldout_facts',held)]:
 (pdir/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
(root/'build_summary.json').write_text(json.dumps({'protocol':PROTOCOL_V2,'config':{'seed':1234,'render_policy':RenderPolicy.fixed_rendering().to_dict()}}))
fields=['arm','probe_set','fact_id','pass','template_idx','truth_value','entity_tier','assigned_frequency','logp_value','logp_competing','schema_version','metric','scoring_policy']
for arm in ['C','R']:
 p=edir/f'beliefs_arm{arm}.csv'
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for rec in inj:
   # same hypothetical model scores in both arms: template0 belief0, template1 belief10
   for ti in ([0,1] if arm=='C' else [0]):
    w.writerow(dict(arm=arm,probe_set='injected',fact_id=rec['fact_id'],pass_='') if False else {'arm':arm,'probe_set':'injected','fact_id':rec['fact_id'],'pass':'cloze_neutral','template_idx':ti,'truth_value':rec['truth_value'],'entity_tier':rec['entity_tier'],'assigned_frequency':rec['assigned_frequency'],'logp_value':-10+10*ti,'logp_competing':-10,'schema_version':2,'metric':'answer_plus_suffix','scoring_policy':'joint_prefix_v2'})
 # contradictory input metadata is also ignored
 p.with_suffix('.meta.json').write_text(json.dumps({'mode':'dry_run_mock_scores' if arm=='R' else 'model_inference','tokenizer':{'tokenizer_sha256':arm*64},'input_hashes':{'probe_sets/injected_facts.jsonl':arm*64},'checkpoint':{'init_seed':0 if arm=='R' else 42}}))
with contextlib.redirect_stdout(io.StringIO()):
 AB.main(['--eval-dir',str(edir),'--cohort','all_injected','--bootstrap','0','--no-tracking'])
rows=list(csv.DictReader((edir/'analysis_v2/tables/paired__R-C__all_injected__cloze_neutral__answer_plus_suffix.csv').open()))
print('root',root,'paired results',[(r['label'],r['n'],r['mean_diff']) for r in rows])
co=json.loads((edir/'analysis_v2/cohorts/all_injected.json').read_text());print('missingness',co['missingness']);print('metadata modes/tokenizers/build hashes/seeds intentionally incompatible: analysis accepted')
