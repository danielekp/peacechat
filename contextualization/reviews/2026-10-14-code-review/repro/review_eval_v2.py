import sys,json
sys.path[:0]=['/home/danielaush/Documents/peacechat/contextualization','/home/danielaush/Documents/peacechat']
from ctx import probes as P,templates as T,cohorts as CH
from ctx.factpool_v2 import build_semantic_pool,split_groups,assign_injection,to_probe_records
from nanochat.tokenizer import RustBPETokenizer
G=build_semantic_pool(600,1234);tr,he=split_groups(G,.3,1234);assign_injection(tr,[1,4,16,64,256],1234)
inj,held=to_probe_records(tr,he,['C','R','X','E']);pol=T.RenderPolicy();cfg={'protocol':'v2-corrected-2026-10','seed':1234,'render_policy':pol.to_dict()}
items,_,_=P.build_items(inj,held,cfg)
bad=[]
for r in inj:
 ins=T.render_inserts(P.FactView(r),0,1234,pol,arms=('X',))['X']
 if len(ins.spans['value'])>1:
  it,_=P.build_wrapper_item(r,cfg);bad.append((r,it))
print('V2 default 600 groups:',len(inj),'injected;',len(bad),'malformed wrapper candidates')
for r,it in bad[:1]: print('example',r['fact_id'],r['value'],r['competing_value'],it.context+it.cand_competing)
fixed=T.RenderPolicy.fixed_rendering();fixed_cfg={'protocol':fixed.protocol,'seed':1234,'render_policy':fixed.to_dict()}
wrong=[]
for r in inj:
 ins=T.render_inserts(P.FactView(r),0,1234,fixed,arms=('R',))['R'];actual=r['surface_forms'][ins.paraphrase_index]
 nominal=CH.realized_training_forms(r,fixed_cfg)
 if nominal[0]!=actual:
  wrong.append((r,actual,nominal[0]))
print('V2 fixed-rendering 600 groups: wrong nominal form',len(wrong),'/',len(inj))
for r,act,nom in wrong[:1]: print('example',r['fact_id'],'ACTUAL',act,'REPORTED',nom)
misclass=0
for r in inj:
 ins=T.render_inserts(P.FactView(r),0,1234,fixed,arms=('R',))['R'];actual=r['surface_forms'][ins.paraphrase_index]
 nominal=CH.realized_training_forms(r,fixed_cfg)
 misclass+=CH.is_verbatim_trained(r,[actual])!=CH.is_verbatim_trained(r,nominal)
print('wrong memorization subgroup',misclass,'/',len(inj))
tok=RustBPETokenizer.from_directory('/home/danielaush/.cache/ctx_wandb_to_hf/ctx_C_d24_base/tokenizer')
merged=[]
for it in items:
 sv,sc,meta=P.tokenize_pair(tok,tok.get_bos_token_id(),it)
 for name,seq,spans,cand in [('v',sv,it.answer_spans_value,it.cand_value),('c',sc,it.answer_spans_competing,it.cand_competing)]:
  ids=seq.ids[1:];lens=P.token_byte_lengths(tok,ids);ends=[];pos=0
  for n in lens: pos+=n;ends.append(pos)
  for (_,te),(_,e) in zip(seq.answer_tok_spans,spans):
   be=len((it.context+cand[:e]).encode())
   if ends[te-2]>be:
    merged.append((name,it,seq,meta,te,be,ends[te-2]))
print('answer/suffix BPE overlap candidates',len(merged))
for name,it,seq,meta,*_ in merged[:1]: print('example',name,it.fact_id,it.context,it.cand_value,it.cand_competing,'meta',meta,'answer tokens',[tok.decode(seq.ids[s:e]) for s,e in seq.answer_tok_spans])
