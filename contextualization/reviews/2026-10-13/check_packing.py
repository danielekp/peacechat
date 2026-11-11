"""Exercise the repository packing function with numpy-backed tensor operations.
This isolates document selection/cropping without requiring a working CUDA torch install.
"""
import ast
from pathlib import Path
import numpy as np

class Tensor:
    def __init__(self, a): self.a = a
    def __getitem__(self, key): return Tensor(self.a[key])
    def __setitem__(self, key, value): self.a[key] = value.a if isinstance(value, Tensor) else value
    def view(self, *shape): return Tensor(self.a.reshape(*shape))
    def copy_(self, other, **kwargs): self.a[...] = other.a
class TorchShim:
    long = np.int64
    @staticmethod
    def empty(shape, **kwargs): return Tensor(np.empty(shape, dtype=np.int64))
    @staticmethod
    def tensor(value, **kwargs): return Tensor(np.array(value, dtype=np.int64))
class Tokenizer:
    def get_bos_token_id(self): return 0
    def encode(self, docs, **kwargs): return docs

source=(Path(__file__).resolve().parents[3] / 'nanochat' / 'dataloader.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='tokenizing_distributed_data_loader_with_state_bos_bestfit')
for arm,length in [('R',2049),('X',2059)]:
    doc=[0]+[1]*(length-2)+[999]
    def batches(*args,**kwargs):
        while True: yield [doc]*128, (0,0,1)
    env={'torch':TorchShim,'_document_batches':batches}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<repository loader>','exec'),env)
    loader=env[node.name](Tokenizer(),1,2048,'train',device='cpu',buffer_size=2)
    x,y,state=next(loader)
    print(arm,'document_tokens',length,'scored_target_occurrences',int((y.a==999).sum()))
print('Reported total training tokens:',8352*8*64*2048)
