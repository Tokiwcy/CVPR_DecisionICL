"""Frozen scope adapter: Flowers102 replaces Caltech; legacy artifacts stay intact."""
from pathlib import Path
import argparse,gc,sys,time
import numpy as np
import run_qwen3_gpu_accelerated_queue as q
r=q.r
BASE=q.BASE
FLOWERS=BASE/'gpu_batch2_flowers102_20260911'
PREP=FLOWERS/'preparation'
DATA=FLOWERS/'data'
DOC=r.PROJECT/'docs/qwen3_flowers_replacement_20260911_zh.md'
ORDER=('dtd','aircraft','cub','dogs','flowers102')
OLD_SOURCE=r.common.source
OLD_CONFIGURE=q.configure
OLD_FAST=q.FAST
q.live.DATASETS=ORDER
q.live.LABELS=('DTD','Aircraft','CUB','Dogs','Flowers102')
r.BATCHES['flowers102']=2

class Flowers102Source(r.common.Source):
    def manifest(self,split):return DATA/(split+'.jsonl')
    def embedding(self,split):return PREP/('siglip.'+split+'.embeddings')

def source(dataset):
    if dataset=='caltech101':raise ValueError('Caltech retired by user; no new execution')
    if dataset!='flowers102':return OLD_SOURCE(dataset)
    return Flowers102Source('flowers102',DATA,PREP/'flowers102/atom_bank.final.json',PREP/'flowers102',PREP/'flowers102',PREP/'flowers102/label_embeddings.npy',PREP/'flowers102/label_order_protocol.json',r.read(DATA/'dataset_config.json')['instruction'])

r.common.source=source

def configure(dataset,shots,fast=None):
    if dataset not in ORDER:raise ValueError('dataset outside user scope')
    q.FAST=FLOWERS if dataset=='flowers102' else OLD_FAST
    root=OLD_CONFIGURE(dataset,shots,fast)
    if dataset=='flowers102':r.source_files=lambda:q.fast_sources()+[DOC]
    return root
q.configure=configure

def cleanup(obj):
    del obj;gc.collect()
    import torch
    torch.cuda.empty_cache()

def freeze(path,files):
    if path.exists():r.verify_files(r.read(path)['files'])
    else:r.write(path,dict(files=r.hashes(files)))

def valid(path):
    if not path.exists():return False
    r.verify_files(r.read(path)['files']);return True

def pair(prefix):return [Path(str(prefix)+ext) for ext in ('.json','.npz')]

def sources():return r.source_files()+[Path(__file__),DOC]

def validate_data():
    marker=FLOWERS/'data_audit.json'
    if valid(marker):return
    seen=set();counts={};files=[DOC,Path(__file__),DATA/'dataset_config.json']
    labels=None
    for split,n in (('train',1019),('validation',1019),('test',6149)):
        rows=r.common.load_manifest(source('flowers102').manifest(split),expected_split=split)
        assert len(rows)==n and len({s.sample_id for s in rows})==n
        hashes={s.image_sha256 for s in rows}
        assert not seen.intersection(hashes),'Flowers102 split leakage'
        seen.update(hashes)
        current=sorted({s.label for s in rows})
        assert len(current)==102
        if labels is None:labels=current
        assert current==labels
        for sample in rows:
            assert r.file_sha256(sample.image_path)==sample.image_sha256,'Flowers102 image drift'
        counts[split]=n;files.append(source('flowers102').manifest(split))
    r.write(marker,dict(files=r.hashes(files),counts=counts,classes=102,image_bytes_verified=True,test_used_for_tuning=False))
    r.emit('flowers102_data_audited',dataset='flowers102',counts=counts)

def smoke():
    marker=FLOWERS/'smoke.freeze.json'
    if valid(marker):return
    backend=q.b.BatchedBackend();reports=[]
    try:
        for shots,family in ((1,'baseline'),(2,'baseline'),(2,'circles')):
            labels,items=q.b.records('flowers102',shots,family,3)
            a=backend.forward_many(items[:2],source('flowers102').instruction,labels)
            repeat=backend.forward_many(items[:2],source('flowers102').instruction,labels)
            assert [x['raw_output'] for x in a]==[x['raw_output'] for x in repeat],'fixed group repeat drift'
            tail=backend.forward_many(items[2:],source('flowers102').instruction,labels)
            assert len(a)==2 and len(tail)==1
            reports.append(dict(shots=shots,family=family,validation_ids=[x['query_id'] for x in items],outputs=a+tail,repeat_equal=True))
        r.write(FLOWERS/'validation_smoke.json',dict(status='passed',validation_only=True,test_manifest_loaded=False,batch_size=2,structural_caption_fixture=True,source_files=r.hashes(sources()),outputs=reports))
        freeze(marker,[FLOWERS/'validation_smoke.json',FLOWERS/'data_audit.json']+sources())
        r.emit('flowers102_validation_smoke_passed',dataset='flowers102')
    finally:
        del backend;gc.collect()
        import torch
        torch.cuda.empty_cache()

def prepare():
    configure('flowers102',1)
    r.held();validate_data();smoke()
    import run_qwen25_nf4_class_equal as engine
    import cars_caltech_atom_format as fmt
    import qwen3_4b_backend as target
    from dgdr.encoders import SiglipEncoder
    from dgdr.cli import _balanced_discovery_subset
    from dgdr.discovery import make_discovery_requests
    from experiment_watchdog import verify_receipt
    SIGLIP=r.PROJECT.parent/'models/siglip2-base-patch16-224'
    engine.OUT=PREP;engine.CONTRACT=target.CONTRACT;engine.MODEL=target.MODEL
    engine.held=r.held;engine.load=r.common.load;engine.instruction=lambda ds:source(ds).instruction
    engine.parse_discovery_response=fmt.compatible_parse
    engine.r.verify=verify_receipt
    engine.emit=lambda stage,**kw:r.emit(stage,**({**kw,'total':18} if stage=='discovering' else kw))
    model_files=[p for model in (SIGLIP,target.MODEL) for p in model.rglob('*') if p.is_file() and p.suffix in ('.json','.safetensors','.txt','.model')]
    freeze(PREP/'source.freeze.json',sources()+model_files+[FLOWERS/'data_audit.json',FLOWERS/'smoke.freeze.json'])
    for split in ('train','validation'):
        encode(split,SiglipEncoder,SIGLIP)
    out=PREP/'flowers102';out.mkdir(parents=True,exist_ok=True)
    if not (out/'discovery_inputs.json').exists():
        train=r.legacy.samples('flowers102','train');chosen=_balanced_discovery_subset(train,12,42)
        labels=sorted({x.label for x in train})
        requests=make_discovery_requests(sample_ids=[x.sample_id for x in chosen],image_paths=[x.image_path for x in chosen],task_instruction=source('flowers102').instruction,class_names=labels,batch_size=4,max_groups_per_perspective=6)
        r.write(out/'discovery_inputs.json',dict(labels=labels,selected=[x.to_dict() for x in chosen],requests=[dict(prompt_id=x.prompt_id,batch_id=x.batch_id,sample_ids=list(x.sample_ids),image_paths=list(x.image_paths),perspective=x.perspective,prompt=x.prompt) for x in requests]))
    freeze(PREP/'run.freeze.json',[PREP/'source.freeze.json',out/'discovery_inputs.json',PREP/'train.embeddings.freeze.json',PREP/'validation.embeddings.freeze.json'])
    if not valid(out/'discovery.complete.json'):
        backend=target.Qwen3Backend()
        try:engine.discover('flowers102',backend)
        finally:
            del backend;gc.collect();engine.torch.cuda.empty_cache()
    if not valid(out/'support.freeze.json'):
        encoder=SiglipEncoder(str(SIGLIP),batch_size=64,dtype='float16')
        try:
            bank=engine.r.load_atom_bank(out/'atom_bank.exact.json')
            _,_,raw=engine.evidence('flowers102','train',bank,encoder,'exact.train.evidence')
            dedup=engine.response_aware_deduplicate(group_ids=list(raw.group_ids),semantic_embeddings=encoder.encode_texts([engine.group_semantic_text(g) for g in bank.groups]),response_fingerprints=engine.make_response_fingerprints(raw,pair_count=4096,seed=42),semantic_threshold=.90,response_threshold=.98)
            bank=engine.filter_bank(bank,dedup.kept_group_ids)
            engine.save_atom_bank(out/'atom_bank.final.json',bank);r.write(out/'response_dedup.json',dedup.to_dict())
            for split in ('train','validation'):engine.evidence('flowers102',split,bank,encoder)
            labels=sorted({x.label for x in r.legacy.samples('flowers102','train')})
            np.save(out/'label_embeddings.npy',encoder.encode_texts([f'a photo of a {label}' for label in labels]))
            r.write(out/'label_order_protocol.json',dict(labels=labels,source='train labels only',calibration_inherited=False))
            freeze(out/'support.freeze.json',[out/'discovery.complete.json',out/'atom_bank.final.json',out/'response_dedup.json',out/'label_embeddings.npy',out/'label_order_protocol.json']+sum([pair(out/f'siglip.{s}.evidence') for s in ('train','validation')],[]))
        finally:
            del encoder;gc.collect();engine.torch.cuda.empty_cache()
    encode('test',SiglipEncoder,SIGLIP)
    if not valid(out/'test_evidence.freeze.json'):
        encoder=SiglipEncoder(str(SIGLIP),batch_size=64,dtype='float16')
        try:engine.evidence('flowers102','test',engine.r.load_atom_bank(out/'atom_bank.final.json'),encoder)
        finally:
            del encoder;gc.collect();engine.torch.cuda.empty_cache()
        freeze(out/'test_evidence.freeze.json',[out/'support.freeze.json',PREP/'test.embeddings.freeze.json']+pair(out/'siglip.test.evidence'))
    old_order=r.ORDER
    try:
        r.ORDER=('flowers102',)
        if not (FLOWERS/'input_audit.json').exists():r.audit_inputs()
        else:r.assert_audit()
    finally:r.ORDER=old_order
    r.emit('flowers102_preparation_complete',dataset='flowers102')

def encode(split,Encoder,model):
    marker=PREP/(split+'.embeddings.freeze.json')
    if valid(marker):return
    rows=r.legacy.samples('flowers102',split);encoder=Encoder(str(model),batch_size=64,dtype='float16');chunks=[]
    try:
        for start in range(0,len(rows),256):
            r.held();chunks.append(encoder.encode_images([x.image_path for x in rows[start:start+256]]))
            r.emit('flowers102_siglip_images',dataset='flowers102',split=split,done=min(start+256,len(rows)),total=len(rows))
        r.common.EmbeddingStore(tuple(x.sample_id for x in rows),np.concatenate(chunks),encoder.model_id,encoder.model_revision).save(source('flowers102').embedding(split))
        freeze(marker,[source('flowers102').manifest(split),PREP/'source.freeze.json']+pair(source('flowers102').embedding(split)))
    finally:
        del encoder;gc.collect()
        import torch
        torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('all','check','report','verify'));p.add_argument('--watch',action='store_true');p.add_argument('--dataset',choices=ORDER);p.add_argument('--shots',type=int,choices=(1,2));args=p.parse_args()
    if args.stage=='report':
        sys.argv=[sys.argv[0]]+(['--watch'] if args.watch else [])
        q.live.main();return
    if args.stage=='verify':q.verify_one(args.dataset,args.shots);return
    if args.stage=='check':
        configure('flowers102',1);validate_data()
        r.write(BASE/'flowers_scope_preflight.json',dict(status='passed',datasets=list(ORDER),files=r.hashes([Path(__file__),DOC,FLOWERS/'data_audit.json']),caltech_retired=True))
        return
    ready=BASE/'remaining_without_cars.complete.json'
    while not ready.exists():
        for root in (BASE,FLOWERS):
            if any((root/n).exists() for n in ('DIAGNOSTIC_HOLD.json','PAUSED_BY_USER.json','WATCHDOG_STOP')):raise RuntimeError('Flowers pending queue held')
        time.sleep(30)
    r.verify_files(r.read(ready)['files'])
    import msvcrt
    lock=(BASE/'gpu_queue.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);
    while True:
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
        except OSError:time.sleep(5)
    r.verify_files(r.read(BASE/'flowers_scope_preflight.json')['files'])
    for d,s in (('dtd',1),('dtd',2)):q.verify_one(d,s)
    configure('flowers102',1)
    locations=r.read(BASE/'execution_locations.json');locations['_active_root']=str(FLOWERS);r.write(BASE/'execution_locations.json',locations)
    prepare()
    for shots in (1,2):q.run_one('flowers102',shots)
    r.ORDER=ORDER;q.final_report()

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if len(sys.argv)>1 and sys.argv[1]=='all':
            r.emit('failed',error_type=type(exc).__name__,error=str(exc))
            if not any((BASE/n).exists() for n in ('DIAGNOSTIC_HOLD.json','PAUSED_BY_USER.json','WATCHDOG_STOP')):r.write(FLOWERS/'DIAGNOSTIC_HOLD.json',dict(reason=str(exc),time=time.time()))
        raise
