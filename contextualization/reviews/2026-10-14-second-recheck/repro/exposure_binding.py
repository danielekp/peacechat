import sys,tempfile,json,csv,io,contextlib,shutil
from pathlib import Path
sys.path[:0]=['/home/danielaush/Documents/peacechat/contextualization','/home/danielaush/Documents/peacechat']
import analyze_beliefs as AB
from ctx.prepack import schedule_exposure
src=Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/ctx_review4_smoke/build')
root=Path(tempfile.mkdtemp(prefix='ctx-fourth-exposure-binding-'))
for name in ['probe_sets','eval','exposure']:
 shutil.copytree(src/name,root/name,ignore=shutil.ignore_patterns('analysis_v2'))
for name in ['build_summary.json','BUILD_COMPLETE.json','manifest.csv']:
 shutil.copy2(src/name,root/name)
def run(label):
 with contextlib.redirect_stdout(io.StringIO()):
  result=AB.main(['--eval-dir',str(root/'eval'),'--probe-dir',str(root/'probe_sets'),'--build-summary',str(root/'build_summary.json'),'--exposure-dir',str(root/'exposure'),'--cohort','all_injected','--bootstrap','0','--no-tracking'])
 cp=result['compatibility']
 rows=list(csv.DictReader((root/'eval/analysis_v2/tables/memorization__all_injected.csv').open()))
 out={'ok':cp['ok'],'warnings':cp['warnings'],'inputs_bound':[s['inputs_bound'] for arms in cp['replicates'].values() for s in arms.values()], 'memorization_R':[{k:r[k] for k in ['replicate','subgroup','arm','n','mean_t0_minus_other']} for r in rows if r['arm']=='R']}
 (root/(label+'.json')).write_text(json.dumps(out,indent=2))
 print(label,json.dumps(out))
run('baseline')
# Recompute actual exposure CSV+summary with the same prepacked dataset under a short horizon.
# Only outputs are written under the isolated copy. All evaluation CSVs/metas and build inputs remain untouched.
s=schedule_exposure(src,'R',B=2,grad_accum_steps=1,num_iterations=1,world_size=1,data_seed=3,out_dir=root/'exposure')
print('new_trace_config',s['config'],'new_trace_statuses',s['status_counts'])
print('preflight_horizon',json.loads((root/'exposure/PREFLIGHT_OK.json').read_text())['num_iterations'])
run('short_trace_with_existing_preflight')
print('ROOT',root)
