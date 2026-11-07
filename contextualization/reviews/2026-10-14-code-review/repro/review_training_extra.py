import os, sys, json, tempfile, resource, subprocess
from pathlib import Path
sys.path.insert(0,'/home/danielaush/Documents/peacechat')
import numpy as np, torch
from nanochat.dataloader import PrepackedRows
# Same 4 scored labels in same batch, split 1 vs 3 valid targets across microbatches.
# Current base_train takes mean on each microbatch then averages these means.
w=torch.tensor(0.3,requires_grad=True)
labels=[torch.tensor([0,-1,-1]), torch.tensor([1,1,1])]
loss=0
for ys in labels:
 logits=torch.stack((w.expand(3), -w.expand(3)),dim=-1)
 loss=loss+torch.nn.functional.cross_entropy(logits,ys,ignore_index=-1)/2
loss.backward(); a=float(w.grad)
w.grad=None
ys=torch.cat(labels); logits=torch.stack((w.expand(6),-w.expand(6)),dim=-1)
big=torch.nn.functional.cross_entropy(logits,ys,ignore_index=-1);big.backward()
print('SAME_SCORED_TOKENS_GRADIENT_ACCUMULATION',{'B1_GA2':a,'B2_GA1':float(w.grad),'difference':abs(a-float(w.grad))})
# Child resource limit to model a normal 1024-fd Linux host; all files tiny.
d=Path(tempfile.mkdtemp(prefix='ctx_many_chunks_')); rows=[]
for i in range(520):
 rf=f'rows_{i:05}.npy'; mf=f'mask_{i:05}.npy';np.save(d/rf,np.array([[1,2]],dtype=np.int32));np.save(d/mf,np.array([[0,1]],dtype=np.uint8));rows.append({'file':rf,'mask_file':mf,'n_rows':1})
(d/'prepack_manifest.json').write_text(json.dumps({'schema_version':1,'T':1,'n_rows':520,'row_files':rows}))
soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE); resource.setrlimit(resource.RLIMIT_NOFILE,(min(1024,hard),hard))
try:
 p=PrepackedRows(d)
 print('MMAP_FD_LIMIT_RESULT','loaded')
except Exception as e:
 print('MMAP_FD_LIMIT_RESULT',type(e).__name__,str(e))
finally:
 resource.setrlimit(resource.RLIMIT_NOFILE,(soft,hard))
