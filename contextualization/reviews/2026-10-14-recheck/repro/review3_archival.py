"""First-upload ordering check with fake HF API; no network."""
import contextlib,os,sys,tempfile,types
from pathlib import Path
sys.path[:0]=[str(Path.cwd()/'contextualization'),str(Path.cwd())]
from ctx import tracking as T
from tests.test_artifacts import make_arm
from huggingface_hub.errors import RepositoryNotFoundError
T._ENV_LOADED=True
os.environ['HF_NAMESPACE']='fake'
class NewRepoHub:
 def __init__(self):self.exists=False;self.calls=[]
 def list_repo_refs(self,*args,**kw):
  self.calls.append('list_repo_refs')
  if not self.exists:raise RepositoryNotFoundError('404: repository not created yet')
  return types.SimpleNamespace(tags=[],branches=[])
 def create_repo(self,*args,**kw):self.calls.append('create_repo');self.exists=True
 def upload_folder(self,*args,**kw):self.calls.append('upload_folder');return types.SimpleNamespace(oid='new')
 def create_tag(self,*args,**kw):self.calls.append('create_tag')
hub=NewRepoHub();T.hf_api=lambda:hub;T.upload_progress=lambda *a,**kw:contextlib.nullcontext()
with tempfile.TemporaryDirectory() as d:
 exp=Path(d);make_arm(exp)
 try:T.hf_upload_checkpoint(exp,'C',4,run_id='newrun')
 except Exception as e:print(type(e).__name__,str(e))
print('CALL_ORDER',hub.calls,'REPO_CREATED',hub.exists)
