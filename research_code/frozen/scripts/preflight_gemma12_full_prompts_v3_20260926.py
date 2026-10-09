"""All prompt families for Gemma12 main table, train/validation only.
Strict discovery parser; one fixed schema retry, identical to mainline budget.
"""
import os,sys,json,time,faulthandler
from dataclasses import asdict
from pathlib import Path
SCRIPT=Path(__file__).resolve(); CLEAN=SCRIPT.parents[1]
sys.path.insert(0,str(CLEAN/'src'))
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import preflight_gemma4_12b_nf4_20260920 as spec
from gemma12_nf4_backend_v2_20260926 import Gemma12Backend,CONTRACT
from run_qwen3_circles_official_prompt_2shot_four import official_content,METHODS
import run_qwen3_circles_five as circles
from dgdr.discovery import make_discovery_requests
from dgdr.atoms import lint_groups
from gemma12_discovery_format_20260926 import compatible_parse, NORMALIZATIONS
ROOT=CLEAN/'artifacts/priority_1shot_ablation_20260920/gemma12_full_prompt_preflight_v3_20260926'
RETRY='\nYour previous response was not valid for the required schema. Return a complete valid JSON object only.'

def held():
    for root in (ROOT,ROOT.parent):
        for name in ('PAUSED_BY_USER.json','DIAGNOSTIC_HOLD.json','WATCHDOG_STOP'):
            if (root/name).exists():raise RuntimeError(f'held: {root/name}')
def emit(stage,**kw):
    d=dict(stage=stage,pid=os.getpid(),time=time.time(),**kw);spec.write(ROOT/'status.json',d);print(json.dumps(d),flush=True)
def text_image(sample,texts):
    return [dict(role='user',content=[dict(type='image',image=sample.image_path) if x=='<IMAGE>' else dict(type='text',text=x) for x in texts])]
def checked(backend,conv,path,limit=512):
    held();v=backend.forward(conv,limit);spec.write(path,v);return v

def main():
    faulthandler.enable();held();ROOT.mkdir(exist_ok=True,parents=True)
    files=[SCRIPT,ROOT/'parser_cpu_audit.json',ROOT/'format_cpu_audit.json',SCRIPT.with_name('gemma12_discovery_format_20260926.py'),Path(spec.__file__),SCRIPT.with_name('gemma12_nf4_backend_v2_20260926.py'),
        Path(circles.__file__),SCRIPT.with_name('run_qwen3_circles_official_prompt_2shot_four.py'),circles.OFFICIAL_SOURCE,
        SCRIPT.with_name('cars_caltech_atom_format.py'),CLEAN/'src/dgdr/discovery.py',CLEAN/'src/dgdr/schema.py',CLEAN/'src/dgdr/atoms.py']
    datasets={}
    for ds in spec.ORDER:
        train,val,question=spec.dataset_inputs(ds)
        assert question.strip()
        files.extend([spec.manifest(ds,s) for s in ('train','validation')])
        datasets[ds]=(train,val,question)
    freeze=dict(contract=CONTRACT,discovery_attempts=2,retry_suffix=RETRY,test_access=False,
        files={str(p):spec.sha(p) for p in files})
    freeze_path=ROOT/'run.freeze.json'
    if freeze_path.exists():assert spec.read(freeze_path)==freeze
    else:spec.write(freeze_path,freeze)
    if '--cpu-check' in sys.argv:
        emit('cpu_contract_passed',datasets=5,shots=[1,2],prompt_families=['baseline','circles','attribute','caption','discovery']);return
    # Validate previously verified official checkpoint bytes before loading.
    download=spec.read(ROOT.parent/'gemma_preflight/download.complete.json')
    for path,digest in download['files'].items():assert spec.sha(path)==digest
    import msvcrt
    handles=[]
    for p in (CLEAN/'artifacts/qwen3_five_dataset_recalibrated_20260910/gpu_queue.lock',ROOT.parent/'gpu_queue.lock'):
        f=p.open('a+b');f.seek(0)
        if not f.read(1):f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);handles.append(f)
    emit('loading_gemma12_nf4_pread');backend=Gemma12Backend();reports=[]
    from PIL import Image
    def area(x):
        with Image.open(x.image_path) as image:return image.width*image.height
    for ds,(train,val,question) in datasets.items():
        out=ROOT/ds;out.mkdir(exist_ok=True)
        queries=sorted(val,key=lambda x:(-area(x),x.sample_id))[:4]
        demos=sorted(train,key=lambda x:(-area(x),x.sample_id))[:2]
        for sample in queries+demos:assert spec.sha(sample.image_path)==sample.image_sha256
        labels=sorted({x.label for x in train});allow={x.lower() for x in labels}
        rows=[]
        for shot in (1,2):
            emit('validation_classification',dataset=ds,shots=shot)
            conv=[official_content(question,x.image_path,[(demos[(i+j)%2].image_path,demos[(i+j)%2].label) for j in range(shot)],labels,METHODS[0]) for i,x in enumerate(queries[:3])]
            pair=checked(backend,conv[:2],out/f'baseline_{shot}.pair.json')
            repeat=checked(backend,conv[:2],out/f'baseline_{shot}.repeat.json')
            tail=checked(backend,conv[2:],out/f'baseline_{shot}.tail.json')
            assert pair['raw']==repeat['raw'],'repeat drift'
            assert all(isinstance(t,str) and t.strip() for t in pair['texts']+tail['texts']),'blank classification content'
            rows.append(dict(shots=shot,repeat_equal=True,strict_valid=[t.strip().lower() in allow for t in pair['texts']+tail['texts']],pair_stats=pair['stats']))
        emit('validation_circles_descriptions',dataset=ds)
        attributes=[];captions=[]
        for begin in (0,2):
            batch=queries[begin:min(begin+2,3)]
            conv=[text_image(x,['Identify the key attributes of the following image that are most relevant to answering the question.','<IMAGE>',f'Question: {question}',"Please list the top 1 key attributes as short phrases in a section named '### Attributes', one per line, ordered from most to least important."]) for x in batch]
            result=checked(backend,conv,out/f'attributes_{begin}.json')
            attrs=[circles.parse_attribute(t) for t in result['texts']];assert all(attrs)
            conv=[text_image(x,[circles.official_instruction(),'<IMAGE>',f"Manipulation Text: Change the attribute '{a}' to a different plausible value. Ensure the modified caption is concise and contains no more than 77 tokens."]) for x,a in zip(batch,attrs)]
            result=checked(backend,conv,out/f'captions_{begin}.json')
            caps=[circles.parse_caption(t) for t in result['texts']];assert all(caps)
            attributes+=attrs;captions+=caps
        conv=[official_content(question,x.image_path,[(d.image_path,d.label) for d in demos],labels,METHODS[5],a,c) for x,a,c in zip(queries,attributes,captions)]
        emit('validation_circles_classification',dataset=ds,shots=2)
        pair=checked(backend,conv[:2],out/'circles.pair.json')
        repeat=checked(backend,conv[:2],out/'circles.repeat.json')
        tail=checked(backend,conv[2:],out/'circles.tail.json')
        assert pair['raw']==repeat['raw']
        assert all(isinstance(t,str) and t.strip() for t in pair['texts']+tail['texts'])
        circle_valid=[t.strip().lower() in allow for t in pair['texts']+tail['texts']]
        req=make_discovery_requests(sample_ids=[x.sample_id for x in queries],image_paths=[x.image_path for x in queries],task_instruction=question,class_names=labels,batch_size=4,max_groups_per_perspective=6)[0]
        failures=[]
        for attempt in (0,1):
            emit('validation_discovery',dataset=ds,attempt=attempt)
            prompt=req.prompt+(RETRY if attempt else '')
            conv=[[dict(role='user',content=[dict(type='image',image=p) for p in req.image_paths]+[dict(type='text',text=prompt)])]]
            result=checked(backend,conv,out/f'discovery_{attempt}.raw.json',3072)
            spec.write(out/f'discovery_{attempt}.request.json',dict(request=asdict(req),actual_prompt=prompt))
            try:
                groups=compatible_parse(result['texts'][0],request=req,source_model=spec.MODEL_ID)
                assert 1<=len(groups)<=6
                errors=[x.to_dict() for x in lint_groups(groups,max_states=6) if x.level=='error']
                if errors:raise ValueError(str(errors))
                break
            except (ValueError,TypeError,KeyError,AssertionError) as error:
                failures.append(dict(attempt=attempt,error=str(error)))
                spec.write(out/'discovery_rejections.json',failures)
                if attempt==1:raise
        report=dict(dataset=ds,baseline=rows,circles_repeat_equal=True,circles_strict_valid=circle_valid,
            discovery_groups=len(groups),discovery_attempt=attempt,discovery_stats=result['stats'],
            test_access=False,used_for_factor_selection=False)
        spec.write(out/'format_normalizations.json',NORMALIZATIONS)
        spec.write(out/'passed.json',report);reports.append(report)
        emit('dataset_preflight_passed',dataset=ds,done=len(reports),total=5)
    artifacts=[freeze_path]+list(ROOT.glob('*/*.json'))
    spec.write(ROOT/'complete.json',dict(status='passed',reports=reports,contract=CONTRACT,
        files={str(p):spec.sha(p) for p in artifacts},test_access=False,new_test_predictions=0))
    emit('full_prompt_preflight_complete',datasets=5)

if __name__=='__main__':
    try:main()
    except Exception as e:
        emit('blocked',error_type=type(e).__name__,error=str(e))
        spec.write(ROOT/'DIAGNOSTIC_HOLD.json',dict(error=str(e),time=time.time()))
        raise
