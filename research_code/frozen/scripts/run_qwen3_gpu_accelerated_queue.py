"""Fixed-batch GPU queue, retaining the already-started DTD 1-shot contract."""
from __future__ import annotations
import argparse,gc,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_qwen3_five_dataset_recalibrated as r
import benchmark_qwen3_target_batches as b
import report_qwen3_live_results as live
BASE=r.ROOT
FAST=BASE/'gpu_batch2'
DOC=r.PROJECT/'docs/qwen3_gpu_batch2_addendum_20260910_zh.md'
BENCH=BASE/'gpu_benchmark_20260910/benchmark.json'
POLICY=BASE/'gpu_execution_policy.json'
ORIGINAL_CONTRACT=r.infer_contract
ORIGINAL_SOURCES=r.source_files
ORIGINAL_MIGRATE=r.migrate
CURRENT={}
BATCH=2

def combined_held():
    for root in {BASE, r.ROOT}:
        if any((root/name).exists() for name in ('PAUSED_BY_USER.json','DIAGNOSTIC_HOLD.json','WATCHDOG_STOP')):
            raise RuntimeError('combined GPU queue held')

def fast_sources():
    return ORIGINAL_SOURCES()+[DOC,POLICY,BENCH]

def fast_contract():
    contract=ORIGINAL_CONTRACT()
    out=FAST/f"{CURRENT['shots']}shot"/CURRENT['dataset']
    return dict(**{k:v for k,v in contract.items() if k!='batch_size'},batch_size=BATCH,
                execution='fixed-contiguous-unique-context-batches-v1',
                ordered_test_plan_sha256=r.file_sha256(out/'test_plan.jsonl'),
                batch_backend_sha256=r.file_sha256(b.__file__),
                execution_runner_sha256=r.file_sha256(__file__),
                policy_sha256=r.file_sha256(POLICY),
                cache_cross_batch_reuse=False)

def configure(dataset,shots,fast=None):
    if fast is None: fast=(dataset,shots)!=('dtd',1)
    CURRENT.update(dataset=dataset,shots=shots,fast=fast)
    r.ROOT=FAST if fast else BASE
    r.RESOURCES=r.ROOT/'resources'
    r.infer_contract=fast_contract if fast else ORIGINAL_CONTRACT
    r.source_files=fast_sources if fast else ORIGINAL_SOURCES
    r.migrate=(lambda *args,**kwargs:[]) if fast else ORIGINAL_MIGRATE
    r.update_evidence=update_evidence
    r.held=combined_held
    return r.ROOT

class FixedBatchBackend(b.BatchedBackend):
    def __init__(self,dataset,shots):
        # Complete fixed partition is independent of which rows were already cached.
        ph=r.stable_hash(r.infer_contract())
        plan,train,query,labels,records,keys=r.make_records(dataset,shots,ph)
        self.all_records=list(records.values())
        self.group_by_key={rec['key']:i//BATCH for i,rec in enumerate(self.all_records)}
        self.ready={}
        self.active_group=None
        self.group_receipts=[]
        super().__init__()
    def forward(self,record,question,labels):
        key=record['key'];group=self.group_by_key[key]
        if self.active_group!=group:
            batch=self.all_records[group*BATCH:(group+1)*BATCH]
            outputs=self.forward_many(batch,question,labels)
            if len(outputs)!=len(batch):raise ValueError('batch output cardinality drift')
            group_keys=[x['key'] for x in batch]
            signature=r.stable_hash(group_keys)
            self.ready={}
            for item,output in zip(batch,outputs):
                output['stats'].update(fixed_group_index=group,fixed_group_signature=signature)
                self.ready[item['key']]=output
            self.active_group=group
        if key not in self.ready:raise ValueError('fixed batch ordering drift')
        return self.ready[key]

def policy():
    benchmark=r.read(BENCH)
    if len(benchmark['reports'])!=15 or not benchmark['validation_only'] or benchmark['test_manifest_loaded']:
        raise ValueError('incomplete validation batch benchmark')
    for row in benchmark['reports']:
        value=row['values']['2']
        if value['status']!='passed' or len(value['outputs'])!=row['count'] or value['peak_mib']>13500:
            raise ValueError('batch2 runtime benchmark failed')
        if any(a['stats']['exact_label_output'] and not c['stats']['exact_label_output'] for a,c in zip(row['values']['1']['outputs'],value['outputs'])):
            raise ValueError('batch2 introduced new invalid output relative to baseline')
    r.write(POLICY,dict(batch_size=2,selected_on='explicit user choice; validation throughput and VRAM checked; no target correctness',
            description_batches=r.BATCHES,baseline_dtd_1shot_preserved=True,
            test_used=False,benchmark_sha256=r.file_sha256(BENCH),
            global_speedup=sum(x['values']['1']['seconds'] for x in benchmark['reports'])/sum(x['values']['2']['seconds'] for x in benchmark['reports']),
            raw_mismatches={batch:sum(sum(a['raw_output']!=c['raw_output'] for a,c in zip(x['values']['1']['outputs'],x['values'][batch]['outputs'])) for x in benchmark['reports']) for batch in ('2','4')}))

def audit():
    policy()
    configure('dtd',2,True)
    r.audit_inputs()
    benchmark=r.read(BENCH)
    r.write(FAST/'validation_smoke.json',dict(status='passed',validation_only=True,test_manifest_loaded=False,
           execution_batch_size=2,source_files=r.hashes(fast_sources()),benchmark_sha256=r.file_sha256(BENCH),
           outputs=[dict(dataset=x['dataset'],shots=x['shots'],family=x['family'],outputs=x['values']['2']['outputs']) for x in benchmark['reports']]))
    locations={f'{d}/{s}':str((BASE if (d,s)==('dtd',1) else FAST)/f'{s}shot'/d) for d in r.ORDER for s in (1,2)}
    locations['_active_root']=str(BASE)
    r.write(BASE/'execution_locations.json',locations)
    r.write(BASE/'gpu_queue_preflight.json',dict(status='passed',input_audit_sha256=r.file_sha256(FAST/'input_audit.json'),
             execution_policy_sha256=r.file_sha256(POLICY),smoke_sha256=r.file_sha256(FAST/'validation_smoke.json')))

def update_evidence(dataset,shots):
    out=r.ROOT/f'{shots}shot'/dataset
    path=r.PROJECT.parent/'docs/iclr_decision_guided_mainline_zh.md'
    marker=f'<!-- QWEN3_GPU_QUEUE_{dataset}_{shots}SHOT_COMPLETE -->'
    text=path.read_text(encoding='utf-8')
    if marker not in text:
        result=r.read(out/'results.json')
        scores='；'.join(f"{m} {v['metrics']['accuracy']*100:.3f}%" for m,v in result['methods'].items())
        entry=f'\n\n{marker}\nQwen3-VL-4B {dataset} {shots}-shot 完整核验通过，target batch={2 if CURRENT["fast"] else 1}。{scores}。Macro与paired rescue/harm、bootstrap CI、McNemar见 `{out.relative_to(r.PROJECT.parent).as_posix()}/results.json`。独立重校准、探索性结果，不据test调参。\n'
        path.write_text(text+entry,encoding='utf-8')
    live.render()

def verify_one(dataset,shots):
    configure(dataset,shots)
    r.verify_complete(dataset,shots)

def run_one(dataset,shots):
    root=configure(dataset,shots)
    locations=r.read(BASE/'execution_locations.json');locations['_active_root']=str(root);r.write(BASE/'execution_locations.json',locations)
    out=root/f'{shots}shot'/dataset
    if (out/'complete.json').exists():
        r.verify_complete(dataset,shots);live.render();return
    r.held()
    # Mother root controls the combined queue too.
    if any((BASE/name).exists() for name in ('PAUSED_BY_USER.json','DIAGNOSTIC_HOLD.json','WATCHDOG_STOP')):
        raise RuntimeError('combined queue held')
    r.prepare_images(dataset)
    if shots==2:r.prepare_captions(dataset)
    r.fit(dataset,shots)
    backend=FixedBatchBackend(dataset,shots) if CURRENT['fast'] else r.official.OfficialBackend()
    try:r.infer(dataset,shots,backend)
    finally:
        del backend;gc.collect()
        import torch;torch.cuda.empty_cache()
    live.render()

def final_report():
    receipts=[]
    for dataset in r.ORDER:
        for shots in (1,2):
            verify_one(dataset,shots)
            receipts.extend([r.ROOT/f'{shots}shot'/dataset/name for name in ('complete.json','independent_audit.json')])
    live.render()
    report=BASE/'report_zh.md';report.write_text((BASE/'live_results_zh.md').read_text(encoding='utf-8'),encoding='utf-8')
    r.write(BASE/'complete.json',dict(files=r.hashes(receipts+[report]),completed_blocks=10,completed_methods=60))
    r.write(BASE/'status.json',dict(stage='queue_complete',time=time.time(),completed_blocks=10))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('audit','all','verify','report'));p.add_argument('--dataset',choices=r.ORDER);p.add_argument('--shots',type=int,choices=(1,2));args=p.parse_args()
    if args.stage=='audit':audit();return
    if args.stage=='verify':
        verify_one(args.dataset,args.shots);live.render();return
    if args.stage=='report':final_report();return
    import msvcrt
    lock=(BASE/'gpu_queue.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if r.read(BASE/'gpu_queue_preflight.json')['status']!='passed':raise ValueError('GPU queue preflight missing')
    configure('dtd',1);r.assert_audit()
    configure('dtd',2);r.assert_audit()
    for dataset in r.ORDER:
        for shots in (1,2):run_one(dataset,shots)
    final_report()
if __name__=='__main__':main()
