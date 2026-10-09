"""Serial Qwen2.5-VL-7B NF4 official-prompt run, with fresh model-specific factors and fits."""
from pathlib import Path
import argparse,gc,json,sys,time,shutil
import numpy as np
import run_qwen3_gpu_accelerated_queue as q
import run_qwen3_flowers_auto_batch as trial
import qwen25_nf4_official_backend as nf4
import run_qwen25_nf4_class_equal as engine
import cars_caltech_atom_format as fmt
from dgdr.encoders import SiglipEncoder
from dgdr.cli import _balanced_discovery_subset
from dgdr.discovery import make_discovery_requests
from experiment_watchdog import verify_receipt
r=q.r
PROJECT=r.PROJECT
PRIOR=PROJECT/'artifacts/qwen3_five_dataset_recalibrated_20260910'
BASE=PROJECT/'artifacts/qwen25_nf4_five_dataset_20260913'
DOC=PROJECT/'docs/qwen25_nf4_five_dataset_20260913_zh.md'
ORDER=('dtd','aircraft','cub','dogs','flowers102')
OLD_SOURCE=trial.f.OLD_SOURCE
ORIGINAL_SOURCES=r.source_files
ORIGINAL_CONTRACT=q.ORIGINAL_CONTRACT
CURRENT={}
SIGLIP=PROJECT.parent/'models/siglip2-base-patch16-224'

class Source(r.common.Source):
    def manifest(self,split):
        if self.dataset=='flowers102':return PRIOR/'gpu_batch2_flowers102_20260911/data'/(split+'.jsonl')
        return OLD_SOURCE(self.dataset).manifest(split)
    def embedding(self,split):
        if self.dataset=='flowers102':return PRIOR/'gpu_batch2_flowers102_20260911/preparation'/('siglip.'+split+'.embeddings')
        return OLD_SOURCE(self.dataset).embedding(split)

def source(dataset):
    prep=BASE/'runs'/dataset/'preparation'
    bank=prep/dataset
    instruction=('Identify the flower species shown in the image.' if dataset=='flowers102' else OLD_SOURCE(dataset).instruction)
    if dataset=='flowers102':instruction=r.read(PRIOR/'gpu_batch2_flowers102_20260911/data/dataset_config.json')['instruction']
    return Source(dataset,prep,bank/'atom_bank.final.json',bank,bank,bank/'label_embeddings.npy',bank/'label_order_protocol.json',instruction)

r.common.source=source
r.common.MODEL=nf4.MODEL
r.DOC=DOC
r.official.OfficialBackend.__init__=nf4.initialize
r.legacy.Generator.__init__=nf4.initialize
r.migrate=lambda *args,**kwargs:[]
# Only prompt formatting helpers are reused; their task instruction is this dataset's.
trial.f.source=lambda unused:source(CURRENT['dataset'])


def held():
    for root in (BASE,r.ROOT):
        if any((root/n).exists() for n in ('PAUSED_BY_USER.json','DIAGNOSTIC_HOLD.json','WATCHDOG_STOP')):raise RuntimeError('NF4 queue held')
r.held=held

def configure(dataset,shots):
    CURRENT.update(dataset=dataset,shots=shots)
    r.ROOT=BASE/'runs'/dataset;r.RESOURCES=r.ROOT/'resources';r.ORDER=(dataset,)
    q.BASE=BASE;q.FAST=r.ROOT;q.CURRENT.update(dataset=dataset,shots=shots,fast=True)
    q.BATCH=2
    selection=r.ROOT/'description_batch_trial/selection.json'
    r.source_files=lambda:ORIGINAL_SOURCES()+[Path(__file__),Path(nf4.__file__),DOC,BASE/'cpu_preflight.json']+([selection] if shots==2 and selection.exists() else [])
    r.infer_contract=contract
    return r.ROOT

def contract():
    value=ORIGINAL_CONTRACT()
    root=r.ROOT;out=root/f"{CURRENT['shots']}shot"/CURRENT['dataset']
    return {**value,'model_id':nf4.MODEL_ID,'quantization':'nf4','double_quant':True,'compute_dtype':'bfloat16','batch_size':2,'placement_policy':nf4.PLACEMENT_CONTRACT,
        'runtime_device_map':r.read(root/'validation_smoke.json')['device_map'],
        'ordered_test_plan_sha256':r.file_sha256(out/'test_plan.jsonl'),
        'nf4_backend_sha256':r.file_sha256(nf4.__file__),'execution_runner_sha256':r.file_sha256(__file__),
        'execution':'fixed-contiguous-unique-context-batches-v1','cache_cross_model_reuse':False}

class FixedBackend(q.FixedBatchBackend):
    def forward_many(self,*args):
        self.torch.cuda.empty_cache()
        return super().forward_many(*args)

def release():
    gc.collect()
    import torch
    torch.cuda.empty_cache()

def pair(prefix):return [Path(str(prefix)+ext) for ext in ('.json','.npz')]

def freeze(path,files):
    if path.exists():r.verify_files(r.read(path)['files'])
    else:r.write(path,dict(files=r.hashes(files)))

def valid(path):
    if not path.exists():return False
    r.verify_files(r.read(path)['files']);return True

def smoke(dataset):
    path=r.ROOT/'validation_smoke.json'
    if path.exists():r.verify_files(r.read(path)['source_files']);return
    backend=q.b.BatchedBackend();reports=[]
    try:
        for shots,family in ((1,'baseline'),(2,'baseline'),(2,'circles')):
            labels,items=q.b.records(dataset,shots,family,3)
            a=backend.forward_many(items[:2],source(dataset).instruction,labels)
            repeat=backend.forward_many(items[:2],source(dataset).instruction,labels)
            tail=backend.forward_many(items[2:],source(dataset).instruction,labels)
            assert len(a)==2 and len(tail)==1
            assert [x['raw_output'] for x in a]==[x['raw_output'] for x in repeat],'fixed batch output drift'
            assert all(x['raw_output'].strip() for x in a+tail),'empty validation output'
            reports.append(dict(shots=shots,family=family,outputs=a,tail=tail,validation_ids=[x['query_id'] for x in items]))
        from PIL import Image
        def largest(split,n):
            ranked=[]
            for x in r.legacy.samples(dataset,split):
                with Image.open(x.image_path) as im:w,h=im.size
                ranked.append((w*h,x.sample_id,x))
            return [v[2] for v in sorted(ranked,key=lambda v:(-v[0],v[1]))[:n]]
        demos,queries=largest('train',2),largest('validation',2)
        for family in ('baseline','circles'):
            labels,items=q.b.records(dataset,2,family,2)
            for item,query in zip(items,queries):
                item.update(query_id=query.sample_id,query_image=query.image_path,demo_ids=[x.sample_id for x in demos],demo_images=[x.image_path for x in demos],demo_labels=[x.label for x in demos])
            backend.torch.cuda.empty_cache();backend.torch.cuda.reset_peak_memory_stats()
            values=backend.forward_many(items,source(dataset).instruction,labels)
            assert len(values)==2 and all(v['raw_output'].strip() for v in values)
            reports.append(dict(shots=2,family=family,stress='largest area train/validation images',outputs=values,peak_reserved_mib=backend.torch.cuda.max_memory_reserved()/2**20))
        r.write(path,dict(status='passed',validation_only=True,test_manifest_loaded=False,device_map=backend.model.hf_device_map,quantization_audit=backend.audit,source_files=r.hashes(r.source_files()),reports=reports))
    finally:del backend;release()

def discovery_smoke(dataset):
    path=r.ROOT/'discovery_gpu_preflight.json'
    if path.exists():r.verify_files(r.read(path)['files']);return
    from PIL import Image
    ranked=[]
    for x in r.legacy.samples(dataset,'validation'):
        with Image.open(x.image_path) as im:w,h=im.size
        ranked.append((w*h,x.sample_id,x))
    chosen=[v[2] for v in sorted(ranked,key=lambda v:(-v[0],v[1]))[:4]]
    labels=sorted({x.label for x in r.legacy.samples(dataset,'train')})
    req=make_discovery_requests(sample_ids=[x.sample_id for x in chosen],image_paths=[x.image_path for x in chosen],task_instruction=source(dataset).instruction,class_names=labels,batch_size=4,max_groups_per_perspective=6)[0]
    r.emit('nf4_discovery_gpu_preflight',dataset=dataset,images=4,validation_only=True)
    backend=nf4.NF4Backend()
    try:
        raw=backend.generate(req.image_paths,req.prompt)
        assert raw.strip()
        r.write(path,dict(status='passed',validation_only=True,used_for_factor_selection=False,raw=raw,stats=backend.last_generation_stats,quantization_audit=backend.audit,files=r.hashes(r.source_files()+[source(dataset).manifest('validation')]+[Path(x.image_path) for x in chosen])))
    finally:del backend;release()


def prepare(dataset):
    root=configure(dataset,1);held();s=source(dataset);prep=root/'preparation';bankdir=prep/dataset
    # Reuse only encoder features covered by the prior completed, verified run.
    counts={};seen=set()
    for split in ('train','validation','test'):
        rows,embedding=r.common.load(dataset,split)
        assert Path(embedding.model_id).resolve()==SIGLIP.resolve(),'SigLIP encoder identity drift'
        images={x.image_sha256 for x in rows}
        assert not seen.intersection(images),'split leakage'
        seen|=images
        assert len(rows)==len({x.sample_id for x in rows})
        for x in rows:assert r.file_sha256(x.image_path)==x.image_sha256
        counts[split]=len(rows)
    smoke(dataset)
    discovery_smoke(dataset)
    model_files=[p for model in (SIGLIP,nf4.MODEL) for p in model.rglob('*') if p.is_file() and p.suffix in ('.json','.safetensors','.txt','.model')]
    freeze(prep/'source.freeze.json',r.source_files()+model_files+[root/'validation_smoke.json',root/'discovery_gpu_preflight.json']+[s.manifest(x) for x in counts]+sum([pair(s.embedding(x)) for x in counts],[]))
    engine.OUT=prep;engine.CONTRACT=nf4.CONTRACT;engine.MODEL=nf4.MODEL
    engine.held=held;engine.load=r.common.load;engine.instruction=lambda ds:source(ds).instruction
    engine.parse_discovery_response=fmt.compatible_parse;engine.r.verify=verify_receipt
    engine.emit=lambda stage,**kw:r.emit(stage,**({**kw,'total':18} if stage=='discovering' else kw))
    if not (bankdir/'discovery_inputs.json').exists():
        train=r.legacy.samples(dataset,'train');chosen=_balanced_discovery_subset(train,12,42)
        labels=sorted({x.label for x in train})
        requests=make_discovery_requests(sample_ids=[x.sample_id for x in chosen],image_paths=[x.image_path for x in chosen],task_instruction=s.instruction,class_names=labels,batch_size=4,max_groups_per_perspective=6)
        r.write(bankdir/'discovery_inputs.json',dict(labels=labels,selected=[x.to_dict() for x in chosen],requests=[dict(prompt_id=x.prompt_id,batch_id=x.batch_id,sample_ids=list(x.sample_ids),image_paths=list(x.image_paths),perspective=x.perspective,prompt=x.prompt) for x in requests]))
    freeze(prep/'run.freeze.json',[prep/'source.freeze.json',bankdir/'discovery_inputs.json'])
    if not valid(bankdir/'discovery.complete.json'):
        backend=nf4.NF4Backend()
        try:engine.discover(dataset,backend)
        finally:del backend;release()
    if not valid(bankdir/'support.freeze.json'):
        encoder=SiglipEncoder(str(SIGLIP),batch_size=64,dtype='float16')
        try:
            bank=engine.r.load_atom_bank(bankdir/'atom_bank.exact.json')
            _,_,raw=engine.evidence(dataset,'train',bank,encoder,'exact.train.evidence')
            dedup=engine.response_aware_deduplicate(group_ids=list(raw.group_ids),semantic_embeddings=encoder.encode_texts([engine.group_semantic_text(g) for g in bank.groups]),response_fingerprints=engine.make_response_fingerprints(raw,pair_count=4096,seed=42),semantic_threshold=.90,response_threshold=.98)
            bank=engine.filter_bank(bank,dedup.kept_group_ids);engine.save_atom_bank(bankdir/'atom_bank.final.json',bank);r.write(bankdir/'response_dedup.json',dedup.to_dict())
            for split in ('train','validation'):engine.evidence(dataset,split,bank,encoder)
            labels=sorted({x.label for x in r.legacy.samples(dataset,'train')})
            np.save(bankdir/'label_embeddings.npy',encoder.encode_texts([f'a photo of a {label}' for label in labels]))
            r.write(bankdir/'label_order_protocol.json',dict(labels=labels,source='train labels only',calibration_inherited=False))
            freeze(bankdir/'support.freeze.json',[bankdir/'discovery.complete.json',bankdir/'atom_bank.final.json',bankdir/'response_dedup.json',bankdir/'label_embeddings.npy',bankdir/'label_order_protocol.json']+sum([pair(bankdir/f'siglip.{x}.evidence') for x in ('train','validation')],[]))
        finally:del encoder;release()
    if not valid(bankdir/'test_evidence.freeze.json'):
        encoder=SiglipEncoder(str(SIGLIP),batch_size=64,dtype='float16')
        try:engine.evidence(dataset,'test',engine.r.load_atom_bank(bankdir/'atom_bank.final.json'),encoder)
        finally:del encoder;release()
        freeze(bankdir/'test_evidence.freeze.json',[bankdir/'support.freeze.json']+pair(bankdir/'siglip.test.evidence'))
    if not (root/'input_audit.json').exists():r.audit_inputs()
    else:r.assert_audit()
    seed_images(dataset)
    r.emit('nf4_dataset_prepared',dataset=dataset,counts=counts)

def seed_images(dataset):
    out=r.RESOURCES/dataset;out.mkdir(parents=True,exist_ok=True)
    receipt=out/'reuse_receipt.json'
    if receipt.exists():r.verify_files(r.read(receipt)['files']);return
    parent=PRIOR/('gpu_batch2_flowers102_20260911' if dataset=='flowers102' else 'gpu_batch2')/'resources'/dataset
    files=[]
    for split in ('train','validation','test'):
        meta=r.read(parent/f'clip.{split}.npz.json')
        assert Path(meta['model_path']).resolve()==r.legacy.CLIP_MODEL.resolve(),'CLIP identity drift'
        assert meta['npz_sha256']==r.file_sha256(parent/f'clip.{split}.npz'),'CLIP cache hash drift'
        r.legacy.load_embeddings(parent/f'clip.{split}.npz',[x.sample_id for x in r.legacy.samples(dataset,split)])
        for ext in ('.npz','.npz.json'):
            src=parent/f'clip.{split}{ext}';dest=out/src.name
            shutil.copyfile(src,dest);assert r.file_sha256(src)==r.file_sha256(dest)
            files.extend([src,dest])
    files.append(BASE/'predecessor_verified.json')
    r.write(receipt,dict(files=r.hashes(files),target_predictions_reused=False,descriptions_reused=False,encoder='same frozen CLIP ViT-g',verified_prior= str(BASE/'predecessor_verified.json')))

def select_description_batch(dataset):
    marker=r.ROOT/'description_batch_trial/selection.json'
    if marker.exists():
        saved=r.read(marker);r.verify_files(saved['files']);r.BATCHES[dataset]=saved['selected_batch'];return
    if any((r.RESOURCES/dataset).glob('counterfactual_descriptions.*.jsonl')):raise RuntimeError('Unbound description cache')
    from PIL import Image
    rows=r.legacy.samples(dataset,'validation');rank=[]
    for x in rows:
        with Image.open(x.image_path) as im:w,h=im.size
        rank.append((w*h,max(w/h,h/w),x.sample_id,x))
    chosen={x[2]:x[3] for x in sorted(rank,key=lambda x:(-x[0],x[2]))[:16]}
    for x in sorted(rank,key=lambda x:(-x[1],x[2])):
        chosen.setdefault(x[2],x[3])
        if len(chosen)==32:break
    items=list(chosen.values());reports=[]
    counts=r.read(r.ROOT/'input_audit.json')['counts'][dataset]
    for size in (2,4,8):
        held();generator=None
        try:
            generator=r.legacy.Generator();trial.describe(generator,items[:size])
            generator.torch.cuda.synchronize();generator.torch.cuda.reset_peak_memory_stats();start=time.perf_counter();outputs=[]
            for i in range(0,32,size):held();outputs+=trial.describe(generator,items[i:i+size])
            generator.torch.cuda.synchronize();seconds=time.perf_counter()-start
            tails={counts[x]%size for x in ('validation','test')}-{0}
            tail_outputs={str(n):trial.describe(generator,items[:n]) for n in tails}
            assert trial.describe(generator,items[:size])==outputs[:size],'repeated outputs differ'
            report=dict(status='passed',batch=size,images_per_minute=1920/seconds,peak_mib=generator.torch.cuda.max_memory_reserved()/2**20,outputs=outputs,tail_outputs=tail_outputs)
        except Exception as exc:report=dict(status='failed',batch=size,error_type=type(exc).__name__,error=str(exc))
        finally:
            if generator is not None:del generator
            release()
        reports.append(report);r.write(marker.parent/f'batch{size}.json',report)
        r.emit('nf4_description_batch_candidate',dataset=dataset,batch=size,status=report['status'])
    passed=[x for x in reports if x['status']=='passed' and x['peak_mib']<=13312]
    if not passed:raise RuntimeError('All NF4 description runtime candidates failed; diagnose before resume')
    speed=max(x['images_per_minute'] for x in passed)
    selected=min((x for x in passed if x['images_per_minute']>=speed*.95),key=lambda x:x['batch'])
    r.write(marker,dict(status='passed',selected_batch=selected['batch'],validation_only=True,test_correctness_used=False,files=r.hashes([Path(__file__),Path(nf4.__file__),DOC,source(dataset).manifest('validation'),r.ROOT/'validation_smoke.json']+[x.image_path for x in items]+[marker.parent/f'batch{x}.json' for x in (2,4,8)])))
    r.BATCHES[dataset]=selected['batch'];r.emit('nf4_description_batch_selected',dataset=dataset,description_batch=selected['batch'])

def render():
    live=q.live;live.ROOT=BASE;live.DATASETS=ORDER;live.LABELS=('DTD','Aircraft','CUB','Dogs','Flowers102')
    live.render()
    p=BASE/'live_results_zh.md';p.write_text(p.read_text(encoding='utf8').replace('Qwen3-VL-4B','Qwen2.5-VL-7B NF4 4bit / BF16 compute'),encoding='utf8')

def update_evidence(dataset,shots):
    marker=f'<!-- QWEN25_NF4_OFFICIAL_FRESH_{dataset}_{shots}SHOT_COMPLETE -->'
    p=PROJECT.parent/'docs/iclr_decision_guided_mainline_zh.md'
    if marker not in p.read_text(encoding='utf8'):
        result=r.read(r.ROOT/f'{shots}shot'/dataset/'results.json')
        scores='；'.join(f"{m} {v['metrics']['accuracy']*100:.3f}%" for m,v in result['methods'].items())
        with p.open('a',encoding='utf8') as out:out.write(f'\n\n{marker}\nQwen2.5-VL-7B NF4 4bit / BF16 compute {dataset} {shots}shot独立因子与校准完整核验通过，target batch2。{scores}。探索性，test不用于调参。配对指标见该模型独立累计表。\n')
    render()
r.update_evidence=update_evidence

def initialize():
    BASE.mkdir(parents=True,exist_ok=True)
    locations={f'{d}/{s}':str(BASE/'runs'/d/f'{s}shot'/d) for d in ORDER for s in (1,2)}
    if not (BASE/'execution_locations.json').exists():r.write(BASE/'execution_locations.json',locations)
    render()

def verify_predecessor():
    ready=PRIOR/'complete.json'
    union=dict(r.read(ready)['files']);union.update(r.hashes([ready]))
    locations=r.read(PRIOR/'execution_locations.json')
    for dataset in ORDER:
        for shots in (1,2):
            out=Path(locations[f'{dataset}/{shots}'])
            audit=r.read(out/'independent_audit.json');assert audit['status']=='passed'
            complete=r.read(out/'complete.json')
            plan=r.read(out/'plan.freeze.json')
            assert plan['independent_recalibration'] and len(plan['methods'])==(5 if shots==1 else 7)
            assert r.file_sha256(out/'test_plan.jsonl')==plan['plan_sha256']
            union.update(audit['files']);union.update(complete['files']);union.update(plan['files'])
            input_audit=r.read(out.parent.parent/'input_audit.json');assert input_audit['status']=='passed'
            union.update(input_audit['files'])
            union.update(r.hashes([out/'plan.freeze.json',out/'test_plan.jsonl']))
    r.verify_files(union)
    receipt=BASE/'predecessor_verified.json'
    value=dict(files=union,completed_blocks=10,completed_methods=60,status='passed')
    if receipt.exists():
        assert r.read(receipt)==value,'predecessor provenance changed'
    else:r.write(receipt,value)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('all','report','cpu-check'));args=parser.parse_args()
    if args.stage=='report':render();return
    if args.stage=='cpu-check':return cpu_check()
    initialize()
    # Wait without holding or competing for the GPU used by Qwen3.
    ready=PRIOR/'complete.json'
    r.ROOT=BASE;r.emit('waiting_for_qwen3_complete',predecessor=str(ready),gpu_loaded=False)
    while not ready.exists():
        r.ROOT=BASE;held();time.sleep(30)
    verify_predecessor()
    import msvcrt
    lock=(PRIOR/'gpu_queue.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    while True:
        r.ROOT=BASE;held()
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
        except OSError:time.sleep(5)
    r.verify_files(r.read(BASE/'cpu_preflight.json')['files'])
    r.ROOT=BASE;r.emit('nf4_queue_started',model=nf4.MODEL_ID)
    for dataset in ORDER:
        prepare(dataset)
        for shots in (1,2):
            root=configure(dataset,shots);held()
            loc=r.read(BASE/'execution_locations.json');loc['_active_root']=str(root);r.write(BASE/'execution_locations.json',loc)
            out=root/f'{shots}shot'/dataset
            if (out/'complete.json').exists():r.verify_complete(dataset,shots);render();continue
            if shots==2:select_description_batch(dataset);r.prepare_captions(dataset)
            r.fit(dataset,shots)
            backend=FixedBackend(dataset,shots)
            try:r.infer(dataset,shots,backend)
            finally:del backend;release()
    render();result=r.read(BASE/'live_results.json');assert result['completed_blocks']==10 and not result['errors']
    receipts=[BASE/'runs'/d/f'{s}shot'/d/name for d in ORDER for s in (1,2) for name in ('complete.json','independent_audit.json')]
    r.write(BASE/'complete.json',dict(files=r.hashes(receipts),model=nf4.MODEL_ID,quantization='nf4',double_quant=True,compute_dtype='bfloat16',completed_blocks=10,completed_methods=60))
    r.ROOT=BASE;r.emit('queue_complete',completed_methods=60)

def cpu_check():
    config=r.read(nf4.MODEL/'config.json');assert config['model_type']=='qwen2_5_vl' and 'quantization_config' not in config
    index=r.read(nf4.MODEL/'model.safetensors.index.json')
    for shard in set(index['weight_map'].values()):assert (nf4.MODEL/shard).is_file()
    json.dumps(nf4.CONTRACT,sort_keys=True)
    assert nf4.CONTRACT['quantization']=='nf4' and nf4.CONTRACT['double_quant'] and nf4.CONTRACT['compute_dtype']=='bfloat16'
    assert len(r.METHODS)==7 and len(ORDER)==5
    assert len({str(BASE/'runs'/d/f'{s}shot'/d) for d in ORDER for s in (1,2)})==10
    assert r.official.OfficialBackend.__init__ is nf4.initialize
    assert r.legacy.Generator.__init__ is nf4.initialize
    assert r.common.MODEL==nf4.MODEL and q.BATCH==2
    counts={};prompt_checks=0
    for dataset in ORDER:
        s=source(dataset);assert s.bank.is_relative_to(BASE) and not s.bank.exists()
        split_rows={split:r.legacy.samples(dataset,split) for split in ('train','validation','test')}
        counts[dataset]={split:len(rows) for split,rows in split_rows.items()}
        seen=set()
        for split,rows in split_rows.items():
            assert len(rows)==len({x.sample_id for x in rows})
            image_hashes={x.image_sha256 for x in rows}
            assert not image_hashes.intersection(seen);seen|=image_hashes
        for shots,family in ((1,'baseline'),(2,'baseline'),(2,'circles')):
            labels,items=q.b.records(dataset,shots,family,3)
            for item in items:
                assert len(item['demo_ids'])==shots and item['query_id'] in {x.sample_id for x in split_rows['validation']}
                content=r.official.official_content(s.instruction,item['query_image'],list(zip(item['demo_images'],item['demo_labels'])),labels,item['method'],item['attribute'],item['modified_caption'])
                assert sum(x['type']=='image' for msg in content for x in msg['content'])==shots+2
                prompt_checks+=1
        if dataset!='flowers102':
            for split in split_rows:
                _,embedding=r.common.load(dataset,split)
                assert Path(embedding.model_id).resolve()==SIGLIP.resolve()
    assert counts=={'dtd':{'train':1874,'validation':1872,'test':1880},'aircraft':{'train':3334,'validation':3333,'test':3333},'cub':{'train':5993,'validation':1000,'test':4794},'dogs':{'train':10726,'validation':1197,'test':8580},'flowers102':{'train':1019,'validation':1019,'test':6149}}
    assert prompt_checks==45
    files=ORIGINAL_SOURCES()+[Path(__file__),Path(nf4.__file__),DOC]+list((PROJECT/'src/dgdr').glob('*.py'))+[nf4.MODEL/'config.json',nf4.MODEL/'model.safetensors.index.json']
    r.write(BASE/'cpu_preflight.json',dict(status='passed',model=nf4.MODEL_ID,quantization='nf4',double_quant=True,compute_dtype='bfloat16',weight_bytes=index['metadata']['total_size'],dataset_counts=counts,validation_prompt_checks=prompt_checks,split_isolation=True,gpu_preflight='deferred until predecessor completes and GPU lock acquired',files=r.hashes(files)))
    initialize();print('NF4 CPU setup passed; GPU checks deferred, no inference launched.')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if len(sys.argv)>1 and sys.argv[1]=='all':
            r.ROOT=BASE;r.emit('failed',error_type=type(exc).__name__,error=str(exc));r.write(BASE/'DIAGNOSTIC_HOLD.json',dict(reason=str(exc),time=time.time()))
        raise
