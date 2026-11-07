"""Capture the actual HF staging set and analyze its restored copy offline."""
import contextlib,io,json,os,shutil,sys,tempfile
from pathlib import Path
ROOT=Path('/home/danielaush/Documents/peacechat');sys.path[:0]=[str(ROOT/'contextualization'),str(ROOT)]
from ctx import tracking as T
import analyze_beliefs as AB
root=Path(tempfile.mkdtemp(prefix='ctx_review4_export_'));local=root/'local';local.mkdir()
source=Path('/tmp/ctx_review4_smoke/build')
for f in ['build_summary.json','manifest.csv','BUILD_COMPLETE.json','shard_hashes.json']:
 if (source/f).exists(): shutil.copy(source/f,local/f)
shutil.copytree(source/'probe_sets',local/'probe_sets')
stage=root/'restored'
T._ENV_LOADED=True;T._HF_NS='offline';T.hf_namespace=lambda:'offline';T.load_env=lambda:[]
os.environ['CTX_TRACKING']='0'
def capture(repo,folder,*a,**kw):
 shutil.copytree(folder,stage)
 return 'https://example.invalid/offline','fakecommit'
T.hf_upload_dir=capture
T.hf_upload_build(local)
shutil.copytree(source/'eval/seed_7',stage/'eval')
args=['--eval-dir',str(stage/'eval'),'--probe-dir',str(stage/'probe_sets'),'--manifest',str(stage/'manifest.csv.gz'),'--bootstrap','0','--no-tracking']
with contextlib.redirect_stdout(io.StringIO()):
 result=AB.main(args)
c=result['compatibility']
print('RESTORED_ARCHIVE',json.dumps({'marker_uploaded':(stage/'BUILD_COMPLETE.json').exists(),'strict_ok':c['ok'],'warnings':c['warnings'],'marker_manifest_sha256':c['inputs']['marker_manifest_sha256'],'inputs_bound':{k:v['inputs_bound'] for k,v in c['replicates']['rep0'].items()}}))
print('ROOT',root)
