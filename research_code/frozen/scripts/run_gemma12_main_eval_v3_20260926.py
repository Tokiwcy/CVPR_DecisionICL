"""Gemma 12B main-table plans and fixed-batch inference. No tuning on test.
CPU check is safe during an existing GPU job. 'all' requires all independent
calibrations and acquires both shared GPU locks before any new GPU work.
"""
import argparse, json, os, time
from audit_gemma12_main_eval_20260926 import audit_plan,audit_results
from pathlib import Path
import numpy as np
import run_gemma12_circles_validation_v3_20260926 as cv
from run_qwen3_circles_official_prompt_2shot_four import official_content, METHODS as OFFICIAL
from dgdr.metrics import accuracy_metrics, paired_comparison

cal=cv.cal;prep=cv.prep;core=cv.core;circles=cv.circles
SCRIPT=Path(__file__).resolve();ROOT=prep.ROOT.parent/'evaluation'
METHODS=('random','rices','muier','rices+matched_decision','muier+matched_decision','circles','circles+matched_decision')
CONTRACT=dict(target=cv.TARGET,methods_1shot=METHODS[:5],methods_2shot=METHODS,
    random_seed=42,random_sampling='numpy default_rng, without replacement per query',
    classification='official_content then Gemma EOS-aware official parser, case-insensitive exact label',
    batch_size=2,batch_order='first unique context occurrence in manifest query order then method order',
    cache_reuse='identical full prompt, image bytes/order, protocol, fixed batch partners only',
    test_used_for_tuning=False)
INVALID='__invalid_exact_output__'

def emit(stage,**kw):
    v=dict(stage=stage,time=time.time(),pid=os.getpid(),**kw);core.write(ROOT/'status.json',v);print(json.dumps(v),flush=True)

def held():
    cv.held()
    for n in ('PAUSED_BY_USER.json','DIAGNOSTIC_HOLD.json','WATCHDOG_STOP'):
        if (ROOT/n).exists():raise RuntimeError('Gemma evaluation held')

def files_for_cal(ds,shot):
    out=cal.ROOT/f'{shot}shot'/ds
    paths=[out/'calibration.freeze.json',out/'independent_audit.json']
    if shot==2:paths.append(cv.ROOT/ds/'calibration.freeze.json')
    return paths

def gate(ds,shot):
    for p in files_for_cal(ds,shot):
        if p.name=='independent_audit.json':
            audit=core.read(p);assert audit['status']=='passed'
            assert audit['calibration_sha256']==core.sha(p.parent/'calibration.json')
            assert audit['plan_sha256']==core.sha(p.parent/'validation_plan.json')
        else:cv.verify_tree(p)

def initialize():
    held();marker=ROOT/'run.freeze.json'
    if core.verified(marker):return
    core.write(ROOT/'contract.json',CONTRACT)
    core.freeze(marker,cal.dependencies()+[ROOT/'contract.json',circles.OFFICIAL_SOURCE])

def clip(ds,split,rows):
    loc=Path(core.read(core.OLD/'execution_locations.json')[ds+'/1'])
    old=core.read(loc/'plan.freeze.json')['files'];p=core.resource(ds)/f'clip.{split}.npz'
    files=[core.common.source(ds).manifest(split),p,Path(str(p)+'.json')]
    for f in files:assert old.get(str(f.resolve()))==core.sha(f),'Historical image feature drift'
    return circles.load_embeddings(p,[x.sample_id for x in rows]),files

def prediction(text,labels):
    allowed={x.strip().lower():x for x in labels};assert len(allowed)==len(labels)
    return allowed.get(text.strip().lower(),INVALID)

def conv(ds,q,demos,labels,method,description=None):
    d=description or {}
    return official_content(core.common.source(ds).instruction,q.image_path,
        [(x.image_path,x.label) for x in demos],labels,OFFICIAL[0] if method in METHODS[:5] else OFFICIAL[5],
        d.get('attribute',''),d.get('modified_caption',''))

def validate_prompt(ds,backend):
    out=ROOT/'resources'/ds;marker=out/'validation_smoke.freeze.json'
    if core.verified(marker):return
    gate(ds,1);gate(ds,2)
    train={x.sample_id:x for x in core.manifest(ds,'train')};val=core.manifest(ds,'validation');labels=sorted({x.label for x in train.values()})
    desc={x['query_id']:x for x in core.read(cv.ROOT/ds/'descriptions.json')};files=[ROOT/'run.freeze.json']
    for shot in (1,2):
        vp=core.read(cal.ROOT/f'{shot}shot'/ds/'validation_plan.json')
        cp=core.read(cv.ROOT/ds/'validation_plan.json') if shot==2 else None
        conversations=[]
        for i,q in enumerate(val[:3]):
            method='circles+matched_decision' if shot==2 and i==1 else 'rices+matched_decision'
            pp=cp if method.startswith('circles') else vp
            ids=pp['methods'][method+'.full'][i]
            for x in [q]+[train[j] for j in ids]:assert core.sha(x.image_path)==x.image_sha256
            conversations.append(conv(ds,q,[train[j] for j in ids],labels,method,desc[q.sample_id]))
        outputs=[]
        for name,part in [('pair',conversations[:2]),('repeat',conversations[:2]),('tail',conversations[2:])]:
            held();v=backend.forward(part,512);p=out/f'validation_{shot}.{name}.json';core.write(p,v);files.append(p);outputs.append(v)
        assert outputs[0]['raw']==outputs[1]['raw'],'Validation batch repeat drift'
        assert all(t.strip() for t in outputs[0]['texts']+outputs[2]['texts'])
    core.freeze(marker,files+files_for_cal(ds,1)+files_for_cal(ds,2))
    emit('validation_prompt_smoke_passed',dataset=ds,test_access=False)

def inputs(ds):
    out=ROOT/'resources'/ds;marker=out/'test_inputs.freeze.json'
    if core.verified(marker):return
    assert core.verified(out/'validation_smoke.freeze.json')
    gate(ds,1);gate(ds,2)
    files=[ROOT/'run.freeze.json',out/'validation_smoke.freeze.json']
    for split in ('train','test'):
        rows,em=core.common.load(ds,split)
        assert em.sample_ids==tuple(x.sample_id for x in rows)
        assert len({x.sample_id for x in rows})==len(rows)
        for x in rows:assert core.sha(x.image_path)==x.image_sha256
        _,fs=clip(ds,split,rows);files+=fs+core.pairfiles(core.common.source(ds).embedding(split))
    core.freeze(marker,files)

def test_descriptions(ds,backend):
    out=ROOT/'resources'/ds;marker=out/'descriptions.freeze.json'
    if core.verified(marker):return
    assert core.verified(out/'test_inputs.freeze.json')
    rows=core.manifest(ds,'test');question=core.common.source(ds).instruction;records=[];files=[out/'test_inputs.freeze.json']
    start=time.time();generated=0
    for i,batch in enumerate(cv.batches(rows)):
        held()
        for x in batch:assert core.sha(x.image_path)==x.image_sha256
        ac=[cv.attribute_conversation(x,question) for x in batch]
        sig=core.stable([core.sha(ROOT/'run.freeze.json'),[(x.sample_id,x.image_sha256) for x in batch],ac])
        ap=out/'description_batches'/f'{i:05d}.attributes.json';fresh=not ap.exists()
        ar=cv.stage_output(ap,sig,ac,backend);attrs=[circles.parse_attribute(t) for t in ar['texts']]
        assert len(attrs)==len(batch) and all(attrs)
        cc=[cv.caption_conversation(x,a) for x,a in zip(batch,attrs)]
        cp=out/'description_batches'/f'{i:05d}.captions.json';fresh=fresh or not cp.exists()
        cr=cv.stage_output(cp,core.stable([sig,core.sha(ap),cc]),cc,backend)
        caps=[circles.parse_caption(t) for t in cr['texts']];assert len(caps)==len(batch) and all(caps)
        records.extend(dict(query_id=x.sample_id,image_sha256=x.image_sha256,attribute=a,modified_caption=c,
            fixed_batch_signature=sig,source_model=cal.MODEL_ID,test_label_used=False) for x,a,c in zip(batch,attrs,caps))
        files += [ap,cp];generated+=len(batch) if fresh else 0
        if i%10==0:
            rate=generated/max(time.time()-start,1)*60
            emit('test_descriptions',dataset=ds,done=len(records),total=len(rows),queries_per_minute=rate,
                eta_minutes=(len(rows)-len(records))/rate if rate else None)
    core.write(out/'descriptions.json',records);core.freeze(marker,files+[out/'descriptions.json'])

def evidence(ds):
    out=ROOT/'resources'/ds;marker=out/'test_evidence.freeze.json'
    if core.verified(marker):return
    assert core.verified(out/'test_inputs.freeze.json')
    cv.verify_tree(prep.ROOT/ds/'train_validation_evidence.freeze.json')
    rows,em=core.common.load(ds,'test');bank=core.load_atom_bank(prep.ROOT/ds/prep.MODE/'atom_bank.final.json')
    encoder=core.SiglipEncoder(str(core.SIGLIP),batch_size=64,dtype='float16')
    try:
        ev=core.build_raw_siglip_evidence(samples=rows,image_embeddings=em,bank=bank,encoder=encoder)
        core.engine.r.validate_alignment(rows,em,ev,bank);ev.save(out/'siglip.test.evidence')
        core.freeze(marker,[out/'test_inputs.freeze.json',prep.ROOT/ds/'train_validation_evidence.freeze.json']+core.pairfiles(out/'siglip.test.evidence'))
    finally:del encoder;core.cleanup()

def caption_embeddings(ds):
    out=ROOT/'resources'/ds;marker=out/'captions.freeze.json'
    if core.verified(marker):return
    assert core.verified(out/'descriptions.freeze.json')
    import torch
    from transformers import CLIPModel,CLIPProcessor
    model=CLIPModel.from_pretrained(circles.CLIP_MODEL,local_files_only=True).to('cuda:0').eval()
    processor=CLIPProcessor.from_pretrained(circles.CLIP_MODEL,local_files_only=True);rows=core.read(out/'descriptions.json');chunks=[]
    try:
        for i in range(0,len(rows),64):
            held();ip=processor(text=[x['modified_caption'] for x in rows[i:i+64]],padding=True,truncation=True,max_length=77,return_tensors='pt').to('cuda:0')
            with torch.inference_mode():v=model.get_text_features(input_ids=ip['input_ids'],attention_mask=ip['attention_mask'])
            chunks.append(circles.clip_feature_tensor(v,torch).float().cpu().numpy())
        p=out/'clip.captions.test.npz';circles.save_embeddings(p,[x['query_id'] for x in rows],np.concatenate(chunks),'modified_caption/test')
        core.freeze(marker,[out/'descriptions.freeze.json',p,Path(str(p)+'.json'),cv.ROOT/'run.freeze.json'])
    finally:del model;core.cleanup()

def plan(ds,shot):
    held();out=ROOT/f'{shot}shot'/ds;marker=out/'plan.freeze.json'
    if core.verified(marker):return
    gate(ds,shot);resource=ROOT/'resources'/ds;cv.verify_tree(resource/'test_evidence.freeze.json')
    train,te=core.common.load(ds,'train');query,qe=core.common.load(ds,'test')
    labels=tuple(sorted({x.label for x in train}));lookup={x:i for i,x in enumerate(labels)};ty=np.asarray([lookup[x.label] for x in train])
    ti,_=clip(ds,'train',train);qi,_=clip(ds,'test',query)
    tv=core.load_ev(ds,prep.MODE,'train');ev=core.common.EvidenceStore.load(resource/'siglip.test.evidence')
    assert ev.sample_ids==qe.sample_ids and ev.bank_hash==tv.bank_hash
    proto=cal.build_class_atom_prototypes(samples=train,evidence=tv);assert proto.labels==labels
    lv=np.load(prep.ROOT/ds/'label_embeddings.npy');fit=core.read(cal.ROOT/f'{shot}shot'/ds/'calibration.json')['methods']
    descriptions={};caps=None
    if shot==2:
        cv.verify_tree(resource/'captions.freeze.json')
        caps=circles.load_embeddings(resource/'clip.captions.test.npz',[x.sample_id for x in query])
        descriptions={x['query_id']:x for x in core.read(resource/'descriptions.json')}
        fit.update(core.read(cv.ROOT/ds/'calibration.json')['methods'])
    rng=np.random.default_rng(42);plans=[]
    for begin in range(0,len(query),64):
        held();stop=min(begin+64,len(query));eb=core.common.EmbeddingStore(qe.sample_ids[begin:stop],qe.embeddings[begin:stop],qe.model_id,qe.model_revision)
        vb=core.common.cub.shared.sliced_evidence(ev,begin,stop)
        ps=qi[begin:stop]@ti.T;mp=core.common.parent_scores(eb,te,ty,lv)
        picks={'rices':core.r.top(ps,shot).tolist(),'muier':core.r.top(mp,shot).tolist()}
        for parent,scores in [('rices',ps),('muier',mp)]:
            f=(core.r.official.single_channel_atom_features(ps,ty,vb,proto,len(labels)) if parent=='rices' else core.common.cub.shared.class_features(eb,vb,te,ty,lv,proto)[0])
            cls=cal.score_classes(f,np.asarray(fit[parent+'+matched_decision']['weights'],np.float32)).argmax(1)
            picks[parent+'+matched_decision']=cal.choose(scores,ty,cls,shot)
        if shot==2:
            cs=caps[begin:stop]@ti.T;f,oi,ci=circles.circle_atom_features(ps,cs,ty,vb,proto,len(labels))
            cls=cal.score_classes(f,np.asarray(fit['circles+matched_decision']['weights'],np.float32)).argmax(1)
            picks['circles']=list(zip(circles.last_argmax(ps),circles.last_argmax(cs)))
            picks['circles+matched_decision']=[(oi[i,c],ci[i,c]) for i,c in enumerate(cls)]
        for i,q in enumerate(query[begin:stop]):
            selected={'random':rng.choice(len(train),size=shot,replace=False).tolist()}
            selected.update({m:pp[i] for m,pp in picks.items()})
            methods={m:[train[int(j)].sample_id for j in selected[m]] for m in METHODS[:5 if shot==1 else 7]}
            assert all(len(v)==shot and (m.startswith('circles') or len(set(v))==shot) for m,v in methods.items())
            d=descriptions.get(q.sample_id,{})
            plans.append(dict(query_id=q.sample_id,methods=methods,attribute=d.get('attribute',''),modified_caption=d.get('modified_caption','')))
    assert [x['query_id'] for x in plans]==[x.sample_id for x in query]
    core.write(out/'test_plan.json',plans)
    core.freeze(marker,[ROOT/'run.freeze.json',out/'test_plan.json',resource/'test_evidence.freeze.json']+files_for_cal(ds,shot)+([resource/'captions.freeze.json'] if shot==2 else []))
    emit('test_plan_frozen',dataset=ds,shots=shot,queries=len(plans),test_labels_used=False)

def infer(ds,shot,backend):
    out=ROOT/f'{shot}shot'/ds
    if core.verified(out/'complete.json'):
        audit_results(ds,shot,ROOT);return
    cv.verify_tree(out/'plan.freeze.json')
    audit=core.read(out/'independent_plan_audit.json')
    assert audit['status']=='passed' and audit['plan_freeze_sha256']==core.sha(out/'plan.freeze.json') and audit['plan_sha256']==core.sha(out/'test_plan.json')
    plans=core.read(out/'test_plan.json')
    train={x.sample_id:x for x in core.manifest(ds,'train')};query={x.sample_id:x for x in core.manifest(ds,'test')}
    image_hashes={x.image_path:x.image_sha256 for x in list(train.values())+list(query.values())}
    labels=sorted({x.label for x in train.values()});ph=core.stable([CONTRACT,core.sha(out/'plan.freeze.json')]);records={};mapping={m:{} for m in METHODS[:5 if shot==1 else 7]}
    for row in plans:
        q=query[row['query_id']]
        for method in mapping:
            demos=[train[j] for j in row['methods'][method]]
            conversation=conv(ds,q,demos,labels,method,row)
            key=core.stable([ph,q.sample_id,conversation,q.image_sha256,[(x.sample_id,x.image_sha256) for x in demos]])
            records.setdefault(key,conversation);mapping[method][q.sample_id]=key
    order=list(records);cache={};cp=out/'context_cache.jsonl'
    if cp.exists():
        for line in cp.read_text(encoding='utf-8').splitlines():
            v=json.loads(line);assert v['key'] not in cache and v['key'] in records and v['protocol']==ph
            cache[v['key']]=v
    start=time.time();new=0
    for i in range(0,len(order),2):
        held();keys=order[i:i+2];sig=core.stable(keys)
        for k in keys:
            if k in cache:assert cache[k]['fixed_batch_signature']==sig
        if all(k in cache for k in keys):continue
        for k in keys:
            for msg in records[k]:
                for content in msg['content']:
                    if content['type']=='image':
                        # Frozen manifest bytes were checked in inputs; verify the actual opened image again.
                        path=content['image'];expected=image_hashes.get(path)
                        assert expected and core.sha(path)==expected
        result=backend.forward([records[k] for k in keys],512)
        rawpath=out/'batches'/f'{i//2:06d}.json'
        rawrec=dict(protocol=ph,keys=keys,signature=sig,output=result)
        if rawpath.exists():assert core.read(rawpath)['output']['raw']==result['raw'],'Partial batch repeat drift'
        else:core.write(rawpath,rawrec)
        for j,k in enumerate(keys):
            if k in cache:
                assert cache[k]['raw_output']==result['raw'][j];continue
            v=dict(key=k,protocol=ph,fixed_batch_signature=sig,raw_output=result['raw'][j],
                parsed_text=result['texts'][j],prediction=prediction(result['texts'][j],labels),batch_result_sha256=core.sha(rawpath))
            core.engine.append_context(cp,v);cache[k]=v;new+=1
        if i%100==0:
            rate=new/max(time.time()-start,1)*60
            emit('target_inference',dataset=ds,shots=shot,done=len(cache),total=len(records),contexts_per_minute=rate,eta_minutes=(len(records)-len(cache))/rate if rate else None)
    truth={qid:x.label for qid,x in query.items()};pred={m:{qid:cache[k]['prediction'] for qid,k in mp.items()} for m,mp in mapping.items()};results={}
    for m,p in pred.items():
        assert set(p)==set(truth)
        results[m]=dict(metrics=accuracy_metrics(list(truth.values()),[p[k] for k in truth]),invalid_exact_output_count=sum(x==INVALID for x in p.values()))
        if '+' in m:results[m]['paired_vs_parent']=paired_comparison(labels=list(truth.values()),baseline_predictions=[pred[m.split('+')[0]][k] for k in truth],method_predictions=[p[k] for k in truth])
    core.write(out/'predictions.json',pred)
    core.write(out/'results.json',dict(dataset=ds,shots=shot,model=cal.MODEL_ID,methods=results,unique_contexts=len(records),test_used_for_tuning=False,inherited_weights=False))
    core.freeze(out/'complete.json',[out/'plan.freeze.json',out/'predictions.json',out/'results.json',cp]+sorted((out/'batches').glob('*.json')))
    audit_results(ds,shot,ROOT)
    emit('target_block_complete_audited',dataset=ds,shots=shot,methods=len(results))

def cpu_check():
    initialize()
    assert prediction('Cat',['cat','dog'])=='cat' and prediction('- cat',['cat','dog'])==INVALID
    for ds in core.ORDER:
        tr=core.manifest(ds,'train');va=core.manifest(ds,'validation');labels=sorted({x.label for x in tr})
        for shot in (1,2):
            for method in METHODS[:5 if shot==1 else 7]:
                co=conv(ds,va[0],tr[:shot],labels,method,dict(attribute='shape',modified_caption='A different shape.'))
                assert sum(x['type']=='image' for x in co[0]['content'])==shot+2
        gate(ds,1);gate(ds,2) if (cv.ROOT/ds/'calibration.freeze.json').exists() else None
    core.write(ROOT/'cpu_check.json',dict(status='passed',script_sha256=core.sha(SCRIPT),test_access=False,
        datasets=5,prompt_structure_and_exact_parser=True,remaining_checks='validation GPU smoke and independent test-plan replay required before inference'))
    print('Gemma evaluation CPU contracts passed; no test loaded or inferred',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['cpu-check','all']);args=ap.parse_args()
    if args.stage=='cpu-check':cpu_check();return
    initialize();a=core.read(ROOT/'cpu_check.json');assert a['status']=='passed' and a['script_sha256']==core.sha(SCRIPT)
    for ds in core.ORDER:gate(ds,1);gate(ds,2)
    import msvcrt
    handles=[]
    for path in (core.OLD/'gpu_queue.lock',core.QUEUE/'gpu_queue.lock'):
        f=path.open('a+b');f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);handles.append(f)
    for ds in core.ORDER:
        backend=cv.Gemma12Backend()
        try:validate_prompt(ds,backend)
        finally:del backend;core.cleanup()
        inputs(ds);evidence(ds);plan(ds,1)
        audit_plan(ds,1,ROOT)
        backend=cv.Gemma12Backend()
        try:infer(ds,1,backend);test_descriptions(ds,backend)
        finally:del backend;core.cleanup()
        caption_embeddings(ds);plan(ds,2)
        audit_plan(ds,2,ROOT)
        backend=cv.Gemma12Backend()
        try:infer(ds,2,backend)
        finally:del backend;core.cleanup()

if __name__=='__main__':main()
