"""Offline migration reproduction. All W&B/HF APIs are mocks; no network requests."""
import base64,hashlib,json,runpy,sys,tempfile,types,contextlib,io
from pathlib import Path
script=Path.cwd()/'contextualization/migrate_wandb_to_hf.py'
for fail_verify in [False,True]:
 deleted=[];uploaded={};backup_deleted=[]
 def md5(b): return base64.b64encode(hashlib.md5(b).digest()).decode()
 class Artifact:
  def __init__(self,version,payload):
   self.version=version;self.name='ctx_C_d24_base:'+version;self.payload=payload
   self.size=len(payload);self.description='distinct checkpoint';self.created_at='2026-10-14'
  def files(self):return [types.SimpleNamespace(name='model_000001.pt',size=len(self.payload),digest=md5(self.payload))]
  def delete(self,**kw):deleted.append(self.version)
 versions=[Artifact('v0',b'old A'),Artifact('v1',b'new B')]
 class WB:
  def artifact_collection(self,*a):return types.SimpleNamespace(artifacts=lambda:versions)
  def run(self,*a):return types.SimpleNamespace(notes='',update=lambda:None)
  def runs(self,*a,**kw):
   return [types.SimpleNamespace(name='ctx_R_d24_backup',id='unrelated-R',delete=lambda **kw:backup_deleted.append('unrelated-R'))]
 class HF:
  def whoami(self):return {'name':'fake'}
  def create_repo(self,*a,**kw):pass
  def create_tag(self,*a,**kw):pass
  def upload_folder(self,**kw):
   for p in Path(kw['folder_path']).rglob('*'):
    if p.is_file():uploaded[str(p.relative_to(kw['folder_path']))]=p.read_bytes()
  def repo_info(self,*a,**kw):
   return types.SimpleNamespace(siblings=[] if fail_verify else [types.SimpleNamespace(rfilename=k,size=len(v),lfs=types.SimpleNamespace(sha256=hashlib.sha256(v).hexdigest())) for k,v in uploaded.items()])
 sys.modules['wandb']=types.SimpleNamespace(Api=WB)
 sys.modules['huggingface_hub']=types.SimpleNamespace(HfApi=HF)
 with tempfile.TemporaryDirectory() as d:
  stage=Path(d)/'ctx_C_d24_base';stage.mkdir();(stage/'model_000001.pt').write_bytes(versions[-1].payload)
  sys.argv=[str(script),'--execute','--only','ctx_C_d24_base','--download-dir',d]
  with contextlib.redirect_stdout(io.StringIO()):
   try:runpy.run_path(str(script),run_name='__main__')
   except SystemExit:pass
 print(json.dumps({'verify_failed':fail_verify,'uploaded_checkpoint':uploaded.get('model_000001.pt',b'').decode(),'deleted_artifact_versions':deleted,'deleted_unrelated_backup_runs':backup_deleted}))
