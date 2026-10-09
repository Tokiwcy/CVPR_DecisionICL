"""Resume remaining tasks with lossless JSON-list envelope compatibility."""
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys
import time
import run_qwen25_nf4_class_equal as run

FREEZE=run.OUT/'list_format_compat.freeze.json'
PROTOCOL=run.r.PROJECT/'docs/qwen25_list_format_compat_20260907_zh.md'
ORIGINAL_PARSE=run.parse_discovery_response
PENDING=('aircraft','dogs')
NORMALIZATIONS=[]


def compatible_parse(raw, *, request, source_model):
    clean=re.sub(r'^```(?:json)?\s*','',raw.strip(),flags=re.IGNORECASE)
    clean=re.sub(r'\s*```$','',clean)
    try: value=json.loads(clean)
    except json.JSONDecodeError: return ORIGINAL_PARSE(raw,request=request,source_model=source_model)
    if not isinstance(value,list): return ORIGINAL_PARSE(raw,request=request,source_model=source_model)
    # Structural wrapping only: preserve every item, state, coordinate and description.
    normalized=json.dumps({'groups':value},ensure_ascii=False)
    groups=ORIGINAL_PARSE(normalized,request=request,source_model=source_model)
    NORMALIZATIONS.append(dict(prompt_id=request.prompt_id,raw_hash=run.stable_hash(raw),
        normalized_hash=run.stable_hash(normalized),operation='wrap root list in groups object',groups=len(groups)))
    return groups


def validate():
    run.r.verify(FREEZE)
    run.r.verify(run.OUT/'run.freeze.json')


def run_dataset(ds):
    run.assert_previous_complete(ds)
    out=run.OUT/ds
    if (out/'audit.complete.json').exists():
        run.r.verify(out/'audit.complete.json')
        return
    if ds not in PENDING: raise RuntimeError('Previously completed dataset is unexpectedly incomplete')
    for stage,marker in (('discovery','discovery_format_compat.complete.json'),('prepare','test.freeze.json'),('target','audit.complete.json')):
        run.held()
        if (out/marker).exists():
            run.r.verify(out/marker); continue
        stamp=time.strftime('%Y%m%d_%H%M%S')
        stdout,stderr=out/f'{stage}_listcompat_{stamp}.stdout.log',out/f'{stage}_listcompat_{stamp}.stderr.log'
        run.emit('stage_starting',dataset=ds,worker=stage,stdout=str(stdout),stderr=str(stderr))
        with stdout.open('w',encoding='utf-8') as fo,stderr.open('w',encoding='utf-8') as fe:
            subprocess.run([sys.executable,'-X','utf8','-u',str(Path(__file__).resolve()),'--worker',stage,'--dataset',ds],
                cwd=run.r.PROJECT.parent,env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'},stdout=fo,stderr=fe,check=True)
        run.r.verify(out/marker)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',choices=('discovery','prepare','target'))
    parser.add_argument('--dataset',choices=PENDING)
    args=parser.parse_args()
    run.held(); validate(); run.torch.set_num_threads(4)
    if args.worker:
        if not args.dataset: parser.error('--worker requires --dataset')
        run.assert_previous_complete(args.dataset)
        out=run.OUT/args.dataset
        if args.worker=='discovery':
            run.parse_discovery_response=compatible_parse
            run.discover(args.dataset,run.Qwen25Backend())
            run.write_json(out/'list_normalizations.json',NORMALIZATIONS)
            run.frozen(out/'discovery_format_compat.complete.json',[out/'discovery.complete.json',out/'list_normalizations.json',FREEZE])
        else:
            run.r.verify(out/'discovery_format_compat.complete.json')
            if args.worker=='prepare':
                encoder=run.SiglipEncoder(str(run.SIGLIP),batch_size=64,dtype='float16')
                run.calibrate(args.dataset,encoder); run.prepare_test(args.dataset,encoder)
            else:
                run.evaluate(args.dataset,run.Qwen25Backend()); run.audit_report(args.dataset)
        return
    for ds in run.ORDER: run_dataset(ds)
    run.write_json(run.OUT/'summary.json',{d:run.r.read_json(run.OUT/d/'results.json') for d in run.ORDER})
    run.frozen(run.OUT/'complete.json',[run.OUT/d/'audit.complete.json' for d in run.ORDER]+
        [run.OUT/d/'discovery_format_compat.complete.json' for d in PENDING]+[run.OUT/'summary.json',FREEZE])
    run.emit('all_complete',datasets=4,methods=5)


if __name__=='__main__':
    try: main()
    except Exception as error:
        run.emit('failed',error_type=type(error).__name__,error=str(error))
        raise
