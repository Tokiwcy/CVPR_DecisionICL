"""Matched source ablation: fresh banks, independent calibration, fixed batches."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import run_qwen3_flowers_replacement_queue as original
import run_qwen25_nf4_class_equal as engine
import qwen3_4b_backend as discovery_backend
import cars_caltech_atom_format as fmt
from dgdr.discovery import DiscoveryRequest, build_discovery_prompt
from dgdr.cli import _balanced_discovery_subset
from dgdr.schema import AtomBank
from dgdr.io import save_atom_bank, load_atom_bank
from dgdr.atoms import lint_groups, collapse_exact_duplicate_groups, filter_bank
from dgdr.encoders import SiglipEncoder
from dgdr.atom_label import build_class_atom_prototypes, score_classes
from raw_siglip_evidence import build_raw_siglip_evidence
from run_qwen3_dataset_plan import paired

r = original.r
common = r.common
SCRIPT = Path(__file__).resolve()
QUEUE = r.PROJECT/'artifacts/priority_1shot_ablation_20260920'
LEGACY = QUEUE/'qwen3_core_sources'
ROOT = QUEUE/'qwen3_decision_generic'
DOC = r.PROJECT/'docs/qwen3_decision_generic_execution_20260920_zh.md'
ORDER = ('dtd','aircraft','cub','dogs','flowers102')
MODES = ('matched_decision','generic_visual')
PARENTS = ('rices','muier')
PERSPECTIVES = ('shape and relative proportions','color and texture','pose and silhouette',
                'local visible differences','visibility and missing visual information','background and environment')
LEADS = {
    'matched_decision': 'Identify visible factors that you yourself rely on when making classification decisions for this task.',
    'generic_visual': 'Identify generic visible properties that describe the supplied images, without prioritizing classification usefulness or your own decision process.',
    'task_discriminative': 'Identify visible properties useful for distinguishing the allowed classes in this task, without referring to your own decision process.',
}
SIGLIP = r.PROJECT.parent/'models/siglip2-base-patch16-224'
OLD = r.PROJECT/'artifacts/qwen3_five_dataset_recalibrated_20260910'
read, write, sha, stable = r.read, r.write, r.file_sha256, r.stable_hash

def emit(stage, **kw):
    v=dict(stage=stage,time=time.time(),pid=os.getpid(),**kw)
    write(ROOT/'status.json',v); print(json.dumps(v,ensure_ascii=False),flush=True)

def held():
    for root in (ROOT,QUEUE):
        if any((root/n).exists() for n in ('PAUSED_BY_USER.json','DIAGNOSTIC_HOLD.json','WATCHDOG_STOP')):
            raise RuntimeError('core source ablation held')

def pairfiles(prefix): return [Path(str(prefix)+ext) for ext in ('.json','.npz')]

def freeze(path, paths):
    if path.exists(): r.verify_files(read(path)['files'])
    else: write(path,dict(files=r.hashes(paths)))

def verified(path):
    if not path.exists(): return False
    r.verify_files(read(path)['files']); return True

def cleanup():
    gc.collect()
    import torch
    torch.cuda.empty_cache()

def method(mode,parent): return f'{parent}+{mode}'

def resource(ds):
    p=Path(read(OLD/'execution_locations.json')[ds+'/1'])
    return p.parents[1]/'resources'/ds

def manifest(ds,split): return common.load_manifest(common.source(ds).manifest(split),expected_split=split)

def prompt(ds,mode,labels,perspective):
    shared=build_discovery_prompt(task_instruction=common.source(ds).instruction,class_names=labels,perspective=perspective,max_groups=6)
    suffix=shared[shared.index('Each factor is one well-defined'):]
    return (LEADS[mode]+'\n\nTask: '+common.source(ds).instruction+'\nAllowed classes: '+', '.join(labels)+
            '\nPerspective for this pass: '+perspective+'\n\nInspect all supplied TRAIN images. Propose at most 6 visible factor groups. '
            'Seek a different axis from the other perspectives. A factor must be reusable across most images in this task, '
            'visually measurable from pixels, and stated without referring to a class name or sample ID.\n\n'+suffix)

def requests(ds,mode):
    train=manifest(ds,'train'); labels=sorted({x.label for x in train})
    selected=_balanced_discovery_subset(train,12,42); out=[]
    for b in range(3):
        rows=selected[b*4:(b+1)*4]
        for perspective in PERSPECTIVES:
            text=prompt(ds,mode,labels,perspective)
            v=dict(batch_id=f'batch-{b:04d}',sample_ids=[x.sample_id for x in rows],image_paths=[x.image_path for x in rows],perspective=perspective,prompt=text)
            out.append(dict(prompt_id=stable(v)[:16],**v))
    return dict(labels=labels,selected=[x.to_dict() for x in selected],requests=out)

def initialize():
    held(); ROOT.mkdir(parents=True,exist_ok=True)
    receipt=ROOT/'run.freeze.json'
    if verified(receipt): return
    # Hash-verify reused image embeddings against their completed historical contract.
    files=[SCRIPT,DOC,SCRIPT.with_name('run_qwen3_core_source_ablation_20260920.py'),LEGACY/'run.freeze.json']
    files+=sorted((r.PROJECT/'src/dgdr').rglob('*.py'))
    files += [Path(m.__file__) for m in (original,r,r.official,r.legacy,common,engine,discovery_backend,fmt,original.q.b)]
    files += [Path(m.__file__) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py' and Path(m.__file__).resolve().is_relative_to(r.PROJECT)]
    files += [SCRIPT.with_name(n) for n in ('raw_siglip_evidence.py','qwen25_nf4_backend.py','run_qwen25_nf4_list_compat.py','run_qwen3_dataset_plan.py')]
    for model in (discovery_backend.MODEL,SIGLIP):
        files += [p for p in model.rglob('*') if p.is_file() and p.suffix in ('.json','.safetensors','.model','.txt')]
    counts={}
    for ds in ORDER:
        s=common.source(ds); old=Path(read(OLD/'execution_locations.json')[ds+'/1'])
        oldhash=read(old/'plan.freeze.json')['files']
        wanted=[]; seen=set(); counts[ds]={}
        for split in ('train','validation','test'):
            data,emb=common.load(ds,split)
            hs={x.image_sha256 for x in data}
            assert len({x.sample_id for x in data})==len(data) and not hs.intersection(seen)
            seen.update(hs);counts[ds][split]=len(data)
            for x in data:
                if sha(x.image_path)!=x.image_sha256: raise ValueError('image hash drift: '+x.sample_id)
            wanted += [s.manifest(split)]+pairfiles(s.embedding(split))
            cp=resource(ds)/f'clip.{split}.npz'
            wanted += [cp,Path(str(cp)+'.json')]
            r.legacy.load_embeddings(cp,[x.sample_id for x in data])
        for p in wanted:
            digest=oldhash.get(str(p.resolve()))
            if digest is None or sha(p)!=digest: raise ValueError('unverified encoder reuse: '+str(p))
        files += wanted
        values={mode:requests(ds,mode) for mode in MODES}
        assert all(len(v['requests'])==18 for v in values.values())
        assert len({stable(v['selected']) for v in values.values()})==1
        # Only the first instruction paragraph differs between source conditions.
        for i in range(18):
            assert len({v['requests'][i]['prompt'].split('\n\n',1)[1] for v in values.values()})==1
        for mode,v in values.items():
            p=ROOT/ds/mode/'discovery_inputs.json';write(p,v);files.append(p)
    write(ROOT/'cpu_preflight.json',dict(status='passed',counts=counts,same_images_budget_schema=True,
          test_used_for_tuning=False,test_images_verified_for_integrity_only=True,model='Qwen3-VL-4B BF16',
          discovery_requests_per_bank=18,discovery_max_new_tokens=3072,source_modes=MODES,
          target_batch_size=2,discovery_train_only=True,independent_calibration=True))
    freeze(receipt,files+[ROOT/'cpu_preflight.json'])
    emit('cpu_contract_passed',datasets=list(ORDER))

def discover(ds, backend):
    for mode in MODES:
        held(); out=ROOT/ds/mode
        if verified(out/'discovery.complete.json'): continue
        inp=read(out/'discovery_inputs.json'); groups=[]; failures=[]
        for i,v in enumerate(inp['requests']):
            req=DiscoveryRequest(**{**v,'sample_ids':tuple(v['sample_ids']),'image_paths':tuple(v['image_paths'])})
            for attempt in range(2):
                held(); p=out/f'discovery_response_{i:02d}_{attempt}.json'
                text=req.prompt+('\nYour previous response was not valid for the required schema. Return a complete valid JSON object only.' if attempt else '')
                h=stable([v,text,sha(ROOT/'run.freeze.json')])
                if p.exists():
                    obj=read(p);assert obj['request_hash']==h;raw=obj['raw']
                else:
                    emit('discovering',dataset=ds,mode=mode,request=i+1,total=18,attempt=attempt)
                    start=time.time();raw=backend.generate(req.image_paths,text)
                    write(p,dict(request_hash=h,raw=raw,seconds=time.time()-start,stats=backend.last_generation_stats))
                try:
                    parsed=fmt.compatible_parse(raw,request=req,source_model=backend.model_id)
                    if not 1<=len(parsed)<=6 or any(x.level=='error' for x in lint_groups(parsed,max_states=6)):
                        raise ValueError('invalid group/state schema')
                    groups.extend(parsed);break
                except (ValueError,TypeError,KeyError) as error:
                    failures.append(dict(request=i,attempt=attempt,error=str(error)))
                    write(out/'discovery_rejections.json',failures)
                    if attempt==1: raise RuntimeError(f'{ds}/{mode}/{i} failed fixed schema retry') from error
        bank=AtomBank(task_name=ds,task_instruction=common.source(ds).instruction,class_names=tuple(inp['labels']),factor_scope='task',groups=tuple(groups),discovery_split='train',discovery_seed=42)
        engine.assert_discovery_is_train_only(bank,manifest(ds,'train'))
        keep,removed=collapse_exact_duplicate_groups(bank.groups)
        save_atom_bank(out/'atom_bank.raw.json',bank);save_atom_bank(out/'atom_bank.exact.json',filter_bank(bank,keep))
        write(out/'format_normalizations.json',fmt.NORMALIZATIONS)
        write(out/'exact_dedup.json',dict(removed=removed))
        freeze(out/'discovery.complete.json',[ROOT/'run.freeze.json',out/'discovery_inputs.json',out/'atom_bank.raw.json',out/'atom_bank.exact.json',out/'exact_dedup.json',out/'format_normalizations.json']+sorted(out.glob('discovery_response_*.json')))
        emit('bank_discovered',dataset=ds,mode=mode,groups=len(keep))

def evidence(ds,mode,split,bank,encoder):
    rows,emb=common.load(ds,split)
    ev=build_raw_siglip_evidence(samples=rows,image_embeddings=emb,bank=bank,encoder=encoder)
    engine.r.validate_alignment(rows,emb,ev,bank)
    ev.save(ROOT/ds/mode/f'siglip.{split}.evidence')
    return ev

def prepare(ds):
    marker=ROOT/ds/'evidence.freeze.json'
    if verified(marker): return
    encoder=SiglipEncoder(str(SIGLIP),batch_size=64,dtype='float16')
    try:
        files=[ROOT/'run.freeze.json']
        labels=sorted({x.label for x in manifest(ds,'train')})
        np.save(ROOT/ds/'label_embeddings.npy',encoder.encode_texts([f'a photo of a {x}' for x in labels]))
        files.append(ROOT/ds/'label_embeddings.npy')
        for mode in MODES:
            held(); out=ROOT/ds/mode;verified(out/'discovery.complete.json')
            bank=load_atom_bank(out/'atom_bank.exact.json')
            train,emb=common.load(ds,'train')
            raw=build_raw_siglip_evidence(samples=train,image_embeddings=emb,bank=bank,encoder=encoder)
            dedup=engine.response_aware_deduplicate(group_ids=list(raw.group_ids),semantic_embeddings=encoder.encode_texts([engine.group_semantic_text(g) for g in bank.groups]),response_fingerprints=engine.make_response_fingerprints(raw,pair_count=4096,seed=42),semantic_threshold=.90,response_threshold=.98)
            bank=filter_bank(bank,dedup.kept_group_ids)
            if not bank.groups: raise ValueError('empty deduplicated bank')
            save_atom_bank(out/'atom_bank.final.json',bank);write(out/'response_dedup.json',dedup.to_dict())
            files += [out/'discovery.complete.json',out/'atom_bank.final.json',out/'response_dedup.json']
            for split in ('train','validation','test'):
                evidence(ds,mode,split,bank,encoder);files+=pairfiles(out/f'siglip.{split}.evidence')
            emit('evidence_prepared',dataset=ds,mode=mode,groups=len(bank.groups),states=sum(len(g.states) for g in bank.groups))
        freeze(marker,files)
    finally:
        del encoder;cleanup()

def load_ev(ds,mode,split):
    ev=common.EvidenceStore.load(ROOT/ds/mode/f'siglip.{split}.evidence')
    assert ev.sample_ids==common.load(ds,split)[1].sample_ids
    assert ev.bank_hash==load_atom_bank(ROOT/ds/mode/'atom_bank.final.json').bank_hash
    return ev

def fit(ds):
    out=ROOT/ds
    if verified(out/'plan.freeze.json'): return
    assert verified(out/'evidence.freeze.json')
    train,te=common.load(ds,'train');val,ve=common.load(ds,'validation')
    labels,ty,vy=common.labels_and_y(train,val);lv=np.load(out/'label_embeddings.npy')
    ti=r.legacy.load_embeddings(resource(ds)/'clip.train.npz',[x.sample_id for x in train])
    vi=r.legacy.load_embeddings(resource(ds)/'clip.validation.npz',[x.sample_id for x in val])
    calibration={};prototypes={}
    for mode in MODES:
        held();tv=load_ev(ds,mode,'train');vv=load_ev(ds,mode,'validation')
        proto=build_class_atom_prototypes(samples=train,evidence=tv);assert proto.labels==labels
        prototypes[mode]=proto
        for parent in PARENTS:
            feature=(r.official.single_channel_atom_features(vi@ti.T,ty,vv,proto,len(labels)) if parent=='rices' else common.cub.shared.class_features(ve,vv,te,ty,lv,proto)[0])
            oof,alpha,weights,score,history=r.official.fit_class_equal(feature,vy,val)
            calibration[method(mode,parent)]=dict(alpha=alpha,weights=weights.tolist(),outer_folds=history,full_fit_class_accuracy=score,oof_class_accuracy=float(np.mean(oof==vy)),independently_fit=True,inherited_weights=False)
            emit('independent_calibration',dataset=ds,mode=mode,parent=parent,alpha=alpha)
            del feature
    write(out/'calibration.json',dict(methods=calibration,objective='validation class accuracy',folds=5,seed=42,test_used=False,inherited_weights=False,files=r.hashes([out/'evidence.freeze.json'])))
    query,qe=common.load(ds,'test');qi=r.legacy.load_embeddings(resource(ds)/'clip.test.npz',[x.sample_id for x in query])
    evs={m:load_ev(ds,m,'test') for m in MODES};plan=[]
    for start in range(0,len(query),64):
        held();stop=min(start+64,len(query));block=common.EmbeddingStore(qe.sample_ids[start:stop],qe.embeddings[start:stop],qe.model_id,qe.model_revision)
        ps=qi[start:stop]@ti.T;mp=common.parent_scores(block,te,ty,lv)
        picks={'rices':r.top(ps,1).tolist(),'muier':r.top(mp,1).tolist()}
        for mode in MODES:
            ev=common.cub.shared.sliced_evidence(evs[mode],start,stop)
            for parent,score in [('rices',ps),('muier',mp)]:
                f=(r.official.single_channel_atom_features(ps,ty,ev,prototypes[mode],len(labels)) if parent=='rices' else common.cub.shared.class_features(block,ev,te,ty,lv,prototypes[mode])[0])
                classes=score_classes(f,np.asarray(calibration[method(mode,parent)]['weights'],np.float32)).argmax(1)
                picks[method(mode,parent)]=r.within(score,ty,classes,1)
        for i,q in enumerate(query[start:stop]):
            plan.append(dict(query_id=q.sample_id,methods={m:[train[int(j)].sample_id for j in pp[i]] for m,pp in picks.items()}))
    write(out/'test_plan.json',plan)
    freeze(out/'plan.freeze.json',[ROOT/'run.freeze.json',out/'evidence.freeze.json',out/'calibration.json',out/'test_plan.json'])
    emit('test_plan_frozen',dataset=ds,queries=len(plan),methods=6)

def record(q, demos):
    return dict(query_id=q.sample_id,query_image=q.image_path,demo_ids=[d.sample_id for d in demos],demo_images=[d.image_path for d in demos],demo_labels=[d.label for d in demos],method=r.METHODS[1],attribute='',modified_caption='')

def smoke(ds,backend):
    p=ROOT/ds/'validation_smoke.json'
    if p.exists():
        assert read(p)['contract']==sha(ROOT/'run.freeze.json');return
    tr=manifest(ds,'train');val=manifest(ds,'validation');labels=sorted({x.label for x in tr})
    records=[record(val[i], [tr[i*17%len(tr)]]) for i in range(3)]
    a=backend.forward_many(records[:2],common.source(ds).instruction,labels)
    b=backend.forward_many(records[:2],common.source(ds).instruction,labels)
    c=backend.forward_many(records[2:],common.source(ds).instruction,labels)
    assert len(a)==2 and len(c)==1 and [v['raw_output'] for v in a]==[v['raw_output'] for v in b]
    write(p,dict(status='passed',contract=sha(ROOT/'run.freeze.json'),validation_ids=[x.sample_id for x in val[:3]],batch2_repeat_equal=True,tail_batch=1,outputs=a+c,test_used=False))

def infer(ds,backend):
    out=ROOT/ds
    if verified(out/'complete.json'): return
    assert verified(out/'plan.freeze.json')
    smoke(ds,backend)
    plan=read(out/'test_plan.json');train={x.sample_id:x for x in manifest(ds,'train')};query={x.sample_id:x for x in manifest(ds,'test')}
    labels=sorted({x.label for x in train.values()});question=common.source(ds).instruction
    ph=stable([sha(out/'plan.freeze.json'),sha(ROOT/'run.freeze.json'),'official-normal-1shot-batch2-fixed-v1',common.PROCESSOR_SETTINGS])
    records={};mapping={}
    for row in plan:
        for m,ids in row['methods'].items():
            rec=record(query[row['query_id']],[train[i] for i in ids])
            key=stable([ph,rec,query[row['query_id']].image_sha256,[train[i].image_sha256 for i in ids]])
            records.setdefault(key,rec);mapping[row['query_id'],m]=key
    ordered=list(records);cache={};cp=out/'context_cache.jsonl'
    if cp.exists():
        for line in cp.read_text(encoding='utf-8').splitlines():
            v=json.loads(line);key=v['key']
            if key not in records or v['contract']!=ph or key in cache: raise ValueError('cache contract/key conflict')
            cache[key]=v
    start_time=time.time();new=0
    for i in range(0,len(ordered),2):
        held();keys=ordered[i:i+2];signature=stable(keys)
        for k in keys:
            if k in cache and cache[k]['fixed_group_signature']!=signature: raise ValueError('batch partner drift')
        if all(k in cache for k in keys):continue
        outputs=backend.forward_many([records[k] for k in keys],question,labels)
        assert len(outputs)==len(keys)
        for k,v in zip(keys,outputs):
            if k in cache:
                if cache[k]['raw_output']!=v['raw_output']: raise ValueError('partial group repeat drift')
                continue
            item=dict(key=k,contract=ph,fixed_group_signature=signature,**v)
            engine.append_context(cp,item);cache[k]=item;new+=1
        if i%100==0:
            emit('target_inference',dataset=ds,done=len(cache),total=len(ordered),new_contexts=new,contexts_per_minute=new/max(time.time()-start_time,1)*60)
    truth={k:q.label for k,q in query.items()};pred={};result={}
    for m in plan[0]['methods']:
        pred[m]={qid:cache[mapping[qid,m]]['prediction'] for qid in query}
        result[m]=dict(metrics=r.accuracy_metrics([truth[k] for k in query],[pred[m][k] for k in query]),invalid_exact_output_count=sum(v=='__invalid_exact_output__' for v in pred[m].values()),demo_same_class_precision=float(np.mean([train[row['methods'][m][0]].label==truth[row['query_id']] for row in plan])))
        if '+' in m:result[m]['paired_vs_parent']=paired(pred[m.split('+')[0]],pred[m],truth)
    # Predeclared source contrasts: matched versus each control for each parent.
    contrasts={}
    for parent in PARENTS:
        for control in MODES[1:]:
            contrasts[f'{parent}:matched_vs_{control}']=paired(pred[method(control,parent)],pred[method('matched_decision',parent)],truth)
    write(out/'predictions.json',pred)
    write(out/'results.json',dict(dataset=ds,shots=1,methods=result,source_contrasts=contrasts,test_used_for_tuning=False,unique_contexts=len(records),inherited_weights=False))
    # Re-read the persisted predictions and independently verify all accuracies and rescue/harm counts.
    saved=read(out/'predictions.json')
    for m,v in result.items():
        assert set(saved[m])==set(truth)
        assert abs(sum(saved[m][k]==truth[k] for k in truth)/len(truth)-v['metrics']['accuracy'])<1e-12
        if '+' in m:
            p=saved[m.split('+')[0]]
            assert sum(p[k]!=truth[k] and saved[m][k]==truth[k] for k in truth)==v['paired_vs_parent']['rescue']
            assert sum(p[k]==truth[k] and saved[m][k]!=truth[k] for k in truth)==v['paired_vs_parent']['harm']
    write(out/'independent_audit.json',dict(status='passed',queries=len(truth),methods=len(result),accuracy_and_paired_counts_recomputed=True))
    freeze(out/'complete.json',[out/'plan.freeze.json',out/'validation_smoke.json',cp,out/'predictions.json',out/'results.json',out/'independent_audit.json'])
    report();emit('dataset_complete',dataset=ds,methods=len(result))
    with (r.PROJECT.parent/'docs/iclr_decision_guided_mainline_zh.md').open('a',encoding='utf-8') as f:
        f.write('\n\n<!-- QWEN3_CORE_SOURCE_'+ds.upper()+'_20260920 -->\nQwen3 1-shot matched来源消融 '+ds+' 完整完成，两来源×两parent及两个parent基线，独立train/validation校准，配对来源对照与原始输出见 '+str(out/'results.json')+'。\n')

def report():
    results={}
    for ds in ORDER:
        p=ROOT/ds/'results.json'
        if (ROOT/ds/'complete.json').exists():
            results[ds]=read(p)
            results[ds]['methods']={k:v for k,v in results[ds]['methods'].items() if 'task_discriminative' not in k}
            results[ds]['source_contrasts']={k:v for k,v in results[ds]['source_contrasts'].items() if 'task_discriminative' not in k}
    # Fixed comparison families: 30 atom-parent tests; 20 matched-control tests.
    families={'parent':[],'source':[]}
    for ds,v in results.items():
        for m,item in v['methods'].items():
            if 'paired_vs_parent' in item:families['parent'].append((ds,m,item['paired_vs_parent']))
        for m,item in v['source_contrasts'].items():families['source'].append((ds,m,item))
    for family,size in [('parent',30),('source',20)]:
        ordered=sorted(families[family],key=lambda x:x[2]['exact_mcnemar_p']);last=0.
        for i,(_,_,v) in enumerate(ordered):
            last=max(last,min(1.,(size-i)*v['exact_mcnemar_p']));v['holm_p_fixed_family']=last
            v['holm_family_size']=size
    write(ROOT/'live_results.json',dict(results=results,completed_datasets=len(results),expected_datasets=5))
    lines=['# Qwen3 核心因子来源消融（1-shot）','', '| 数据集 | RICES | MUIER | R+Decision | M+Decision | R+Generic | M+Generic |','|---|---|---|---|---|---|---|']
    methods=list(PARENTS)+[method(mode,parent) for mode in MODES for parent in PARENTS]
    for ds in ORDER:
        lines.append('| '+ds+' | '+' | '.join(f'{results[ds]["methods"][m]["metrics"]["accuracy"]*100:.3f}' if ds in results else '—（未完成）' for m in methods)+' |')
    lines+=['','完整 JSON 包含 parent rescue/harm、matched-vs-control 配对差异及固定比较族 Holm 校正；test 不用于校准。其余消融已按用户指令取消；DTD引用旧完整测量，不重跑。']
    (ROOT/'live_results_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def migrate_discovery():
    """Reuse only identical train-only requests; never import target caches."""
    marker=ROOT/'migration.freeze.json'
    if verified(marker): return
    oldfreeze=read(LEGACY/'run.freeze.json');r.verify_files(oldfreeze['files'])
    newfreeze=read(ROOT/'run.freeze.json')
    for path,digest in oldfreeze['files'].items():
        if path in newfreeze['files'] and newfreeze['files'][path]!=digest:
            raise ValueError('shared source/model/data contract drift: '+path)
    migrated=[];files=[LEGACY/'run.freeze.json',ROOT/'run.freeze.json']
    for ds in ORDER:
        for mode in MODES:
            olddir=LEGACY/ds/mode;newdir=ROOT/ds/mode
            inp=read(newdir/'discovery_inputs.json')
            if not (olddir/'discovery_inputs.json').exists():continue
            assert inp==read(olddir/'discovery_inputs.json'),'discovery request drift'
            for i,v in enumerate(inp['requests']):
                for attempt in range(2):
                    src=olddir/f'discovery_response_{i:02d}_{attempt}.json'
                    if not src.exists():continue
                    obj=read(src)
                    text=v['prompt']+('\nYour previous response was not valid for the required schema. Return a complete valid JSON object only.' if attempt else '')
                    assert obj['request_hash']==stable([v,text,sha(LEGACY/'run.freeze.json')])
                    dst=newdir/src.name
                    value={**obj,'request_hash':stable([v,text,sha(ROOT/'run.freeze.json')]),'migration_source':str(src),'migration_source_sha256':sha(src),'original_request_hash':obj['request_hash']}
                    if dst.exists():assert read(dst)==value
                    else:write(dst,value)
                    migrated.append(dict(source=str(src),destination=str(dst),raw_sha256=stable(obj['raw'])))
                    files.extend([src,dst])
    assert verified(LEGACY/'dtd/complete.json')
    # DTD is a completed historical block, referenced with its original plan and batch contract.
    for name in ('results.json','complete.json'):
        src=LEGACY/'dtd'/name;dst=ROOT/'dtd'/name
        if dst.exists():assert dst.read_bytes()==src.read_bytes()
        else:dst.write_bytes(src.read_bytes())
        files.extend([src,dst])
    write(ROOT/'migration_receipt.json',dict(train_only_responses=migrated,original_dtd_result_reference=str(LEGACY/'dtd/results.json'),target_cache_imported=False,new_target_plan_required=True,shared_dependencies_verified=True))
    freeze(marker,files+[ROOT/'migration_receipt.json'])
    emit('migration_verified',response_files=len(migrated),dtd_referenced=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('all','cpu-check','report'));args=p.parse_args()
    if args.stage=='report':report();return
    initialize()
    migrate_discovery()
    if args.stage=='cpu-check': return
    import msvcrt
    locks=[]
    # Shared with the established Qwen execution queue; also guard this task root.
    for path in (OLD/'gpu_queue.lock',QUEUE/'gpu_queue.lock'):
        handle=path.open('a+b');handle.seek(0);handle.write(b'0');handle.flush();handle.seek(0)
        msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1);locks.append(handle)
    for ds in ORDER:
        held()
        if verified(ROOT/ds/'complete.json'):continue
        if not all((ROOT/ds/m/'discovery.complete.json').exists() for m in MODES):
            emit('loading_discovery_model',dataset=ds)
            backend=discovery_backend.Qwen3Backend()
            try:discover(ds,backend)
            finally:del backend;cleanup()
        prepare(ds);fit(ds)
        emit('loading_target_model',dataset=ds)
        backend=original.q.b.BatchedBackend()
        try:infer(ds,backend)
        finally:del backend;cleanup()
    report();emit('core_source_ablation_complete',datasets=5)

if __name__=='__main__':
    try:main()
    except Exception as error:
        emit('failed',error_type=type(error).__name__,error=str(error))
        write(ROOT/'DIAGNOSTIC_HOLD.json',dict(reason=str(error),time=time.time()))
        raise
