"""Gemma CIRCLES validation descriptions and independently fitted retrieval.
No test manifest or target predictions are read or produced by this entry.
"""
import argparse, os, time, sys
from pathlib import Path
import numpy as np
import calibrate_gemma12_main_v3_20260926 as cal
import run_qwen3_circles_five as circles
from gemma12_nf4_backend_v2_20260926 import Gemma12Backend, CONTRACT as TARGET
from preflight_gemma12_full_prompts_v3_20260926 import text_image

core=cal.core;prep=cal.prep
SCRIPT=Path(__file__).resolve();ROOT=prep.ROOT.parent/'circles_validation'
CONTRACT=dict(target=TARGET,split='validation',description_batch=2,max_new_tokens=512,
    attribute_parser='CIRCLES existing parse_attribute',caption_parser='CIRCLES existing parse_caption',
    clip_model=str(circles.CLIP_MODEL),clip_text_batch=64,clip_max_length=77,
    retrieval='1 original image + 1 counterfactual caption; duplicate demo allowed across channels',
    calibration=cal.CONTRACT,source='Gemma generated descriptions and Gemma train-only Decision bank',test_access=False)

def emit(stage,**kw):
    v=dict(stage=stage,time=time.time(),pid=os.getpid(),**kw)
    core.write(ROOT/'status.json',v);print(core.json.dumps(v),flush=True)

def held():
    cal.held()
    for n in ('PAUSED_BY_USER.json','DIAGNOSTIC_HOLD.json','WATCHDOG_STOP'):
        if (ROOT/n).exists():raise RuntimeError('CIRCLES validation held')

def verify_tree(path,seen=None):
    seen=set() if seen is None else seen;path=Path(path).resolve()
    if path in seen:return
    assert core.verified(path),str(path);seen.add(path)
    for name in core.read(path)['files']:
        p=Path(name)
        if p.suffix=='.json' and ('freeze' in p.name or p.name.endswith('complete.json')):
            verify_tree(p,seen)

def attribute_conversation(sample,question):
    return text_image(sample,['Identify the key attributes of the following image that are most relevant to answering the question.',
        '<IMAGE>',f'Question: {question}',"Please list the top 1 key attributes as short phrases in a section named '### Attributes', one per line, ordered from most to least important."])

def caption_conversation(sample,attribute):
    return text_image(sample,[circles.official_instruction(),'<IMAGE>',
        f"Manipulation Text: Change the attribute '{attribute}' to a different plausible value. Ensure the modified caption is concise and contains no more than 77 tokens."])

def batches(rows):return [rows[i:i+2] for i in range(0,len(rows),2)]

def initialize():
    held();marker=ROOT/'run.freeze.json'
    if core.verified(marker):return
    files=cal.dependencies()+[circles.OFFICIAL_SOURCE]
    for ds in core.ORDER:
        train=core.manifest(ds,'train');validation=core.manifest(ds,'validation')
        assert len({x.sample_id for x in validation})==len(validation)
        for x in validation:assert core.sha(x.image_path)==x.image_sha256
        _,a=cal.clip_inputs(ds,'train',train);_,b=cal.clip_inputs(ds,'validation',validation)
        files+=a+b
    files += [p for p in circles.CLIP_MODEL.rglob('*') if p.is_file() and p.suffix in ('.json','.txt','.model','.safetensors','.bin')]
    core.write(ROOT/'contract.json',CONTRACT);files.append(ROOT/'contract.json')
    core.freeze(marker,files)

def stage_output(path,signature,conversations,backend):
    held()
    if path.exists():
        value=core.read(path);assert value['signature']==signature,'Description cache signature drift'
        return value['output']
    value=backend.forward(conversations,512)
    core.write(path,dict(signature=signature,output=value)) # Raw before parsing.
    return value

def describe(ds,backend):
    held();out=ROOT/ds;marker=out/'descriptions.complete.json'
    if core.verified(marker):return
    validation=core.manifest(ds,'validation');question=core.common.source(ds).instruction
    records=[];files=[ROOT/'run.freeze.json'];start=time.time();generated=0
    for i,batch in enumerate(batches(validation)):
        held()
        for x in batch:assert core.sha(x.image_path)==x.image_sha256
        conv=[attribute_conversation(x,question) for x in batch]
        identity=dict(contract=CONTRACT,run_sha=core.sha(ROOT/'run.freeze.json'),
            ordered_images=[(x.sample_id,x.image_sha256) for x in batch],attribute_prompt=conv)
        sig=core.stable(identity);ap=out/'batches'/f'{i:05d}.attributes.json'
        was_cached=ap.exists();ar=stage_output(ap,sig,conv,backend)
        attrs=[circles.parse_attribute(t) for t in ar['texts']]
        assert len(attrs)==len(batch) and all(attrs),'Empty/short attributes'
        conv=[caption_conversation(x,a) for x,a in zip(batch,attrs)]
        csig=core.stable([identity,core.sha(ap),conv]);cp=out/'batches'/f'{i:05d}.captions.json'
        cap_cached=cp.exists();cr=stage_output(cp,csig,conv,backend)
        caps=[circles.parse_caption(t) for t in cr['texts']]
        assert len(caps)==len(batch) and all(caps),'Empty/short captions'
        files += [ap,cp]
        for x,a,c,rawa,rawc in zip(batch,attrs,caps,ar['raw'],cr['raw']):
            records.append(dict(query_id=x.sample_id,image_sha256=x.image_sha256,attribute=a,
                modified_caption=c,attribute_raw=rawa,caption_raw=rawc,fixed_batch_signature=sig,
                source_model=cal.MODEL_ID,test_label_used=False))
        if not (was_cached and cap_cached):generated+=len(batch)
        if i%10==0 or len(records)==len(validation):
            rate=generated/max(time.time()-start,1)*60
            emit('circles_validation_descriptions',dataset=ds,done=len(records),total=len(validation),
                new_queries=generated,queries_per_minute=rate,
                eta_minutes=(len(validation)-len(records))/rate if rate else None)
    assert [x['query_id'] for x in records]==[x.sample_id for x in validation]
    core.write(out/'descriptions.json',records)
    core.freeze(marker,files+[out/'descriptions.json'])
    emit('circles_validation_descriptions_complete',dataset=ds,queries=len(records),test_access=False)

def embed(ds):
    held();out=ROOT/ds;marker=out/'caption_embeddings.freeze.json'
    if core.verified(marker):return
    verify_tree(out/'descriptions.complete.json')
    import torch
    from transformers import CLIPModel,CLIPProcessor
    torch.set_num_threads(4)
    model=CLIPModel.from_pretrained(circles.CLIP_MODEL,local_files_only=True).to('cuda:0').eval()
    processor=CLIPProcessor.from_pretrained(circles.CLIP_MODEL,local_files_only=True)
    rows=core.read(out/'descriptions.json');chunks=[]
    try:
        for i in range(0,len(rows),64):
            held();batch=rows[i:i+64]
            inp=processor(text=[x['modified_caption'] for x in batch],padding=True,truncation=True,max_length=77,return_tensors='pt').to('cuda:0')
            with torch.inference_mode():value=model.get_text_features(input_ids=inp['input_ids'],attention_mask=inp['attention_mask'])
            chunks.append(circles.clip_feature_tensor(value,torch).float().cpu().numpy())
        p=out/'clip.captions.validation.npz'
        circles.save_embeddings(p,[x['query_id'] for x in rows],np.concatenate(chunks),'modified_caption/validation')
        core.freeze(marker,[out/'descriptions.complete.json',p,Path(str(p)+'.json')])
    finally:del model;core.cleanup()

def fit(ds):
    held();out=ROOT/ds;marker=out/'calibration.freeze.json'
    if core.verified(marker):return
    verify_tree(out/'caption_embeddings.freeze.json')
    verify_tree(prep.ROOT/ds/'train_validation_evidence.freeze.json')
    train=core.manifest(ds,'train');val=core.manifest(ds,'validation')
    labels,ty,vy=core.common.labels_and_y(train,val)
    ti,_=cal.clip_inputs(ds,'train',train);vi,_=cal.clip_inputs(ds,'validation',val)
    ci=circles.load_embeddings(out/'clip.captions.validation.npz',[x.sample_id for x in val])
    tv=core.load_ev(ds,prep.MODE,'train');vv=core.load_ev(ds,prep.MODE,'validation')
    bank=core.load_atom_bank(prep.ROOT/ds/prep.MODE/'atom_bank.final.json')
    assert all(g.source_model==cal.MODEL_ID for g in bank.groups)
    proto=cal.build_class_atom_prototypes(samples=train,evidence=tv);assert proto.labels==labels
    original=vi@ti.T;composed=ci@ti.T
    features,oi,ci=circles.circle_atom_features(original,composed,ty,vv,proto,len(labels))
    result,oof,folds=cal.independent_fit(features,vy,val)
    pred=cal.score_classes(features,np.asarray(result['weights'],np.float32)).argmax(1)
    index=np.arange(len(val))
    pairs={'circles':np.stack([circles.last_argmax(original),circles.last_argmax(composed)],1),
        'circles+matched_decision.oof':np.stack([oi[index,oof],ci[index,oof]],1),
        'circles+matched_decision.full':np.stack([oi[index,pred],ci[index,pred]],1)}
    result.update(bank_hash=bank.bank_hash,source_model=cal.MODEL_ID,shots=2,parent='circles')
    core.write(out/'calibration.json',dict(dataset=ds,shots=2,contract=CONTRACT,
        methods={'circles+matched_decision':result},folds=folds.tolist(),test_used=False,inherited_weights=False))
    core.write(out/'validation_plan.json',dict(query_ids=[x.sample_id for x in val],
        methods={m:[[train[int(j)].sample_id for j in row] for row in pp] for m,pp in pairs.items()},test_access=False))
    core.freeze(marker,[out/'caption_embeddings.freeze.json',prep.ROOT/ds/'train_validation_evidence.freeze.json',
        ROOT/'run.freeze.json',out/'calibration.json',out/'validation_plan.json'])
    emit('circles_validation_calibration_complete',dataset=ds,shots=2,test_access=False)

def cpu_check():
    initialize()
    assert [len(x) for x in batches(list(range(5)))]==[2,2,1]
    a=np.asarray([[1,1,0,3]],np.float32);b=np.asarray([[4,0,2,2]],np.float32);y=np.asarray([0,0,1,1])
    f,oi,ci=circles.circle_class_data(a,b,y,2)
    assert oi.tolist()==[[1,3]] and ci.tolist()==[[0,3]] and np.allclose(f,[[2.5,2.5]])
    for ds in core.ORDER:
        rows=core.manifest(ds,'validation');q=core.common.source(ds).instruction
        assert attribute_conversation(rows[0],q)[0]['content'][1]['type']=='image'
        assert len(caption_conversation(rows[0],'shape')[0]['content'])==3
    core.write(ROOT/'cpu_check.json',dict(status='passed',script_sha256=core.sha(SCRIPT),
        datasets=5,queries=sum(len(core.manifest(ds,'validation')) for ds in core.ORDER),
        fixed_batch_checks=True,circles_tie_and_class_channel_checks=True,test_access=False))
    print('CPU check passed: validation-only CIRCLES prompts, frozen inputs, batch groups and channel/tie selection',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['cpu-check','all'])
    ap.add_argument('--wait-preparation',action='store_true');args=ap.parse_args()
    if args.stage=='cpu-check':cpu_check();return
    initialize();audit=core.read(ROOT/'cpu_check.json')
    assert audit['status']=='passed' and audit['script_sha256']==core.sha(SCRIPT)
    verify_tree(prep.PREFLIGHT/'complete.json')
    while not all((prep.ROOT/ds/'train_validation_evidence.freeze.json').exists() for ds in core.ORDER):
        held()
        if not args.wait_preparation:raise RuntimeError('Gemma preparation incomplete')
        emit('waiting_preparation',completed=sum((prep.ROOT/ds/'train_validation_evidence.freeze.json').exists() for ds in core.ORDER),total=5)
        time.sleep(30)
    for ds in core.ORDER:verify_tree(prep.ROOT/ds/'train_validation_evidence.freeze.json')
    import msvcrt
    handles=[]
    for p in (core.OLD/'gpu_queue.lock',core.QUEUE/'gpu_queue.lock'):
        f=p.open('a+b');f.seek(0)
        while True:
            held()
            try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:
                if not args.wait_preparation:raise
                time.sleep(10);f.seek(0)
        handles.append(f)
    for ds in core.ORDER:
        held()
        if not core.verified(ROOT/ds/'descriptions.complete.json'):
            backend=Gemma12Backend()
            try:describe(ds,backend)
            finally:del backend;core.cleanup()
        embed(ds);fit(ds)
    emit('circles_validation_all_complete',datasets=5,test_access=False)

if __name__=='__main__':
    try:main()
    except Exception as e:
        emit('failed',error_type=type(e).__name__,error=str(e))
        core.write(ROOT/'DIAGNOSTIC_HOLD.json',dict(error=str(e),time=time.time()))
        raise
