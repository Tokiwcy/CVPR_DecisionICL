"""Authorized validation-only description batch selection before Flowers 2-shot."""
from pathlib import Path
import gc,sys,time
import run_qwen3_flowers_replacement_queue as f
r=f.r
DOC=r.PROJECT/'docs/qwen3_flowers_auto_description_batch_20260912_zh.md'
TRIAL=f.FLOWERS/'description_batch_trial'
RECEIPT=TRIAL/'selection.json'
BASE_CONFIGURE=f.configure
BASE_RUN=f.q.run_one


def configure(dataset,shots,fast=None):
    root=BASE_CONFIGURE(dataset,shots,fast)
    if dataset=='flowers102':
        original=r.source_files
        extra=[Path(__file__),DOC]
        if shots==2 and RECEIPT.exists():extra.append(RECEIPT)
        r.source_files=lambda:original()+extra
    return root
f.configure=configure
f.q.configure=configure


def choose(reports):
    valid=[x for x in reports if x['status']=='passed' and x['peak_mib']<=13312]
    if not valid:raise RuntimeError('No description batch passed validation runtime checks')
    best=max(x['images_per_minute'] for x in valid)
    return min((x for x in valid if x['images_per_minute']>=best*.95),key=lambda x:x['batch'])


def describe(generator,batch):
    question=f.source('flowers102').instruction
    raw=generator.generate([(x.image_path,[
        'Identify the key attributes of the following image that are most relevant to answering the question.',
        '<IMAGE>',f'Question: {question}',
        "Please list the top 1 key attributes as short phrases in a section named '### Attributes', one per line, ordered from most to least important."
    ]) for x in batch])
    if len(raw)!=len(batch):raise ValueError('attribute cardinality mismatch')
    attrs=[r.legacy.parse_attribute(x) for x in raw]
    if any(not x for x in attrs):raise ValueError('empty attribute')
    captions=generator.generate([(x.image_path,[r.legacy.official_instruction(),'<IMAGE>',
        f"Manipulation Text: Change the attribute '{a}' to a different plausible value. Ensure the modified caption is concise and contains no more than 77 tokens."])
        for x,a in zip(batch,attrs)])
    if len(captions)!=len(batch) or any(not r.legacy.parse_caption(x) for x in captions):raise ValueError('invalid caption')
    return [dict(query_id=x.sample_id,attribute_response=a,caption_response=c) for x,a,c in zip(batch,raw,captions)]


def selected_images():
    from PIL import Image
    rows=r.legacy.samples('flowers102','validation')
    sizes=[]
    for row in rows:
        with Image.open(row.image_path) as im:
            w,h=im.size
        sizes.append((w*h,max(w/h,h/w),row.sample_id,row))
    # Stress large images and extreme aspect ratios without using class labels.
    large=sorted(sizes,key=lambda x:(-x[0],x[2]))[:16]
    extreme=sorted(sizes,key=lambda x:(-x[1],x[2]))
    chosen={x[2]:x[3] for x in large}
    for x in extreme:
        chosen.setdefault(x[2],x[3])
        if len(chosen)==32:break
    assert len(chosen)==32
    return list(chosen.values())


def select_batch():
    configure('flowers102',2)
    if RECEIPT.exists():
        v=r.read(RECEIPT);r.verify_files(v['files'])
        assert v['status']=='passed' and v['selected_batch'] in (2,4,8)
        r.BATCHES['flowers102']=v['selected_batch'];return
    resources=f.FLOWERS/'resources/flowers102'
    if any(resources.glob('counterfactual_descriptions.*.jsonl')) or (f.FLOWERS/'2shot/flowers102/plan.freeze.json').exists():
        raise RuntimeError('Existing unbound Flowers 2-shot resources require contract review')
    rows=selected_images()
    r.emit('flowers_description_batch_validation',dataset='flowers102',shots=2,candidates=[2,4,8],validation_only=True)
    reports=[]
    for batch_size in (2,4,8):
        r.held()
        generator=None
        try:
            generator=r.legacy.Generator()
            torch=generator.torch
            # Warm up, then measure the same 32 validation images for each batch.
            describe(generator,rows[:batch_size])
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter();outputs=[]
            for start in range(0,len(rows),batch_size):
                r.held();outputs+=describe(generator,rows[start:start+batch_size])
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            # Check both production tail sizes from frozen split counts (no test loading).
            tails={1019%batch_size,6149%batch_size}-{0}
            tail_outputs={str(n):describe(generator,rows[:n]) for n in sorted(tails)}
            repeat=describe(generator,rows[:batch_size])
            if repeat!=outputs[:batch_size]:raise ValueError('Repeated fixed batch outputs differ')
            peak=torch.cuda.max_memory_reserved()/2**20
            item=dict(status='passed',batch=batch_size,seconds=seconds,images_per_minute=32*60/seconds,peak_mib=peak,outputs=outputs,tail_outputs=tail_outputs)
        except Exception as exc:
            # Failure disqualifies this candidate; production has no silent fallback.
            item=dict(status='failed',batch=batch_size,error_type=type(exc).__name__,error=str(exc))
        finally:
            if generator is not None:
                del generator
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        reports.append(item);r.write(TRIAL/f'batch{batch_size}.json',item)
        r.emit('flowers_description_batch_candidate',dataset='flowers102',shots=2,**{k:v for k,v in item.items() if k not in ('outputs','tail_outputs')})
    chosen=choose(reports)
    bound=[Path(__file__),DOC,f.source('flowers102').manifest('validation'),f.FLOWERS/'smoke.freeze.json']
    bound += [x.image_path for x in rows]+[TRIAL/f'batch{n}.json' for n in (2,4,8)]
    r.write(RECEIPT,dict(status='passed',selected_batch=chosen['batch'],validation_only=True,test_correctness_used=False,selection_rule='fastest passing within 13GiB reserved; smaller batch within 5 percent of best speed',files=r.hashes(bound)))
    r.BATCHES['flowers102']=chosen['batch']
    r.emit('flowers_description_batch_selected',dataset='flowers102',shots=2,description_batch=chosen['batch'],target_batch=2)


def run_one(dataset,shots):
    if dataset=='flowers102' and shots==2:
        one=f.FLOWERS/'1shot/flowers102'
        r.verify_files(r.read(one/'complete.json')['files'])
        r.verify_files(r.read(one/'independent_audit.json')['files'])
        select_batch()
    return BASE_RUN(dataset,shots)
f.q.run_one=run_one

if __name__=='__main__':
    try:f.main()
    except Exception as exc:
        if len(sys.argv)>1 and sys.argv[1]=='all':
            r.emit('failed',error_type=type(exc).__name__,error=str(exc))
            if not any((root/name).exists() for root in (f.BASE,f.FLOWERS) for name in ('PAUSED_BY_USER.json','DIAGNOSTIC_HOLD.json','WATCHDOG_STOP')):
                r.write(f.FLOWERS/'DIAGNOSTIC_HOLD.json',dict(reason=str(exc),time=time.time()))
        raise
