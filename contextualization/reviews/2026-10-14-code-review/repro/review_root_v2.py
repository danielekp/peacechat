from pathlib import Path
import ast, argparse, contextlib, sys, tempfile, types, json
sys.path.insert(0, str(Path.cwd()/'contextualization'))
from run_arm import build_train_command
from ctx import tracking as T
from torch.distributed.run import parse_args as launcher_parse_args

# Parse real launcher argv and the actual base_train parser, without executing training.
cmd = build_train_command(24,'dummy',2,init_seed=7,data_seed=11,data_loader='prepacked')
launch = launcher_parse_args(cmd[3:])
tree = ast.parse(Path('scripts/base_train.py').read_text())
parser_nodes = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='parser' for t in node.targets):
        parser_nodes.append(node)
    elif isinstance(node, ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='parser':
        parser_nodes.append(node)
scope={'argparse':argparse}
exec(compile(ast.Module(body=parser_nodes,type_ignores=[]),'base_train_parser','exec'),scope)
print('DDP_CHILD_ARGV',launch.training_script_args)
try:
    scope['parser'].parse_args(launch.training_script_args)
    print('DDP_PARSER_OK')
except SystemExit as e:
    print('DDP_PARSER_EXIT',e.code)

# Offline fake Hub state; no network. Exercise the real upload helper for an old run's eval.
class FakeHub:
    def __init__(self):
        self.commits={'old-A':{'model_000100.pt':'A'},'new-B':{'model_000100.pt':'B'}}
        self.head='new-B'; self.tags={'wandb-A':'old-A','wandb-B':'new-B'}
    def create_repo(self,*a,**kw): pass
    def upload_folder(self,**kw):
        files=self.commits[self.head].copy()
        for p in Path(kw['folder_path']).rglob('*'):
            if p.is_file(): files[str(p.relative_to(kw['folder_path']))]=p.read_text()
        self.head='eval-A-on-B';self.commits[self.head]=files
        return types.SimpleNamespace(oid=self.head)
    def create_tag(self,*a,tag,revision,**kw):
        if tag in self.tags: raise ValueError('tag exists')
        self.tags[tag]=revision
    def delete_tag(self,*a,tag,**kw): del self.tags[tag]
hub=FakeHub(); T.hf_api=lambda:hub; T.hf_private=lambda:True
T.upload_progress=lambda *a,**kw:contextlib.nullcontext()
with tempfile.TemporaryDirectory() as d:
    f=Path(d)/'beliefs_armR.csv';f.write_text('evaluation of checkpoint A')
    T.hf_upload_files('fake/ctx_R_d24_base',{'eval/beliefs_armR.csv':f},tags=('wandb-A',))
print('OLD_RUN_TAG_AFTER_EVAL',json.dumps({'wandb-A':hub.tags['wandb-A'],'tagged_contents':hub.commits[hub.tags['wandb-A']]}))
print('MIGRATION_VERSION_SORT',sorted(['v0','v1','v2','v9','v10'],key=lambda x:x))
