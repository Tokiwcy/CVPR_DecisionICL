"""Gemma12 main table v7: same narrow parser policy; recursive source gates first.
V5 and all prior frozen sources/results remain immutable. DTD audited blocks are
hash-bound imports; unfinished blocks run in this isolated output root.
"""
import argparse, json, os, time
from pathlib import Path
import run_gemma12_main_eval_v5_20260926 as frozen
import gemma12_nf4_backend_v4_20260928 as backend_v4

base=frozen.audited.base
cv=base.cv;core=base.core;cal=base.cal;prep=base.prep
SCRIPT=Path(__file__).resolve()
ROOT=prep.ROOT.parent/'evaluation_v7_role_only_invalid_20260928'
V5_ROOT=prep.ROOT.parent/'evaluation_v5_batched_descriptions'
POLICY=backend_v4.POLICY
BASE_AUDIT_RESULTS=base.audit_results
METHODS=base.METHODS
DATASETS=core.ORDER


def source_block(ds,shot):
    folder=V5_ROOT/f'{shot}shot'/ds
    expected=METHODS[:5 if shot==1 else 7]
    complete=core.read(folder/'complete.json')
    assert core.verified(folder/'complete.json'),f'v5 complete source hashes invalid: {folder}'
    assert core.verified(folder/'plan.freeze.json'),f'v5 plan freeze invalid: {folder}'
    result=core.read(folder/'results.json');pred=core.read(folder/'predictions.json')
    audit=core.read(folder/'independent_audit.json');plan_audit=core.read(folder/'independent_plan_audit.json')
    assert audit.get('status')=='passed' and plan_audit.get('status')=='passed'
    assert audit.get('complete_sha256')==core.sha(folder/'complete.json')
    assert audit.get('results_sha256')==core.sha(folder/'results.json')
    assert audit.get('predictions_sha256')==core.sha(folder/'predictions.json')
    assert audit.get('cache_provenance',{}).get('complete') is True
    assert audit.get('queries')==len(core.manifest(ds,'test'))
    assert plan_audit.get('test_labels_used') is False
    assert plan_audit.get('plan_freeze_sha256')==core.sha(folder/'plan.freeze.json')
    assert all(m in result.get('methods',{}) and m in pred and m in audit.get('methods',{}) for m in expected)
    for method in expected:
        checked=audit['methods'][method];saved=result['methods'][method]
        assert checked['metrics']==saved['metrics']
        assert checked['invalid']==saved['invalid_exact_output_count']
        assert set(pred[method])=={x.sample_id for x in core.manifest(ds,'test')}
        if '+' in method:assert checked['paired_vs_parent']==saved['paired_vs_parent']
    return folder,complete,audit,plan_audit,expected


def verify_import_sources():
    ROOT.mkdir(parents=True,exist_ok=True)
    blocks={};source_files=[]
    for shot in (1,2):
        folder,complete,audit,plan_audit,methods=source_block('dtd',shot)
        rel=f'{shot}shot/dtd'
        blocks[rel]={
            'source_root':str(V5_ROOT.resolve()),'source_complete_sha256':core.sha(folder/'complete.json'),
            'source_results_sha256':core.sha(folder/'results.json'),'source_predictions_sha256':core.sha(folder/'predictions.json'),
            'source_audit_sha256':core.sha(folder/'independent_audit.json'),
            'source_plan_sha256':core.sha(folder/'test_plan.json'),'source_plan_freeze_sha256':core.sha(folder/'plan.freeze.json'),
            'method_count':len(methods),'query_count':audit['queries'],
            'cached_contexts':audit['cache_provenance'].get('cached_contexts'),
            'raw_batches':audit['cache_provenance'].get('raw_batches'),
            'independent_audit_status':'passed','test_labels_used_for_tuning':False
        }
        source_files.extend(Path(p) for p in complete['files'])
        source_files.extend([folder/'complete.json',folder/'results.json',folder/'predictions.json',folder/'independent_audit.json',folder/'plan.freeze.json',folder/'independent_plan_audit.json',folder/'test_plan.json'])
    manifest={'version':'gemma12_v7_imports_20260928','blocks':blocks,'v5_source_root':str(V5_ROOT.resolve()),'immutable_source_results':True}
    manifest_path=ROOT/'imported_blocks.json';freeze_path=ROOT/'imported_blocks.freeze.json'
    if manifest_path.exists():
        assert core.read(manifest_path)==manifest,'Imported DTD source manifest drift'
    else:core.write(manifest_path,manifest)
    unique=sorted({Path(p).resolve() for p in source_files+[manifest_path]})
    if freeze_path.exists():
        assert core.verified(freeze_path),'Imported source freeze no longer verifies'
        assert core.read(freeze_path).get('files'), 'Imported source freeze empty'
    else:core.freeze(freeze_path,unique)
    assert core.verified(freeze_path)
    return manifest,freeze_path


def setup():
    size=frozen.accelerated.configure()
    base.ROOT=ROOT;base.SCRIPT=SCRIPT
    manifest,import_freeze=verify_import_sources()
    base.CONTRACT=dict(base.CONTRACT,description_batch=size,
        circles_validation_root=str(cv.ROOT),description_contract=cv.CONTRACT,
        speed_selection_sha256=core.sha(frozen.accelerated.SPEED/'selection.json'),
        target_parser_recovery=POLICY,
        imported_blocks_manifest_sha256=core.sha(ROOT/'imported_blocks.json'),
        imported_blocks_freeze_sha256=core.sha(import_freeze),
        retrieval_and_calibration_unchanged=True,test_used_for_tuning=False)
    base.audit_plan=frozen.replay.audit_plan
    backend_cls=backend_v4.install();cv.Gemma12Backend=backend_cls
    base.validate_prompt=validate_prompt_v7
    base.audit_results=audit_results_v7
    return manifest,import_freeze


def validate_prompt_v7(ds,backend):
    out=ROOT/'resources'/ds;out.mkdir(parents=True,exist_ok=True)
    marker=out/'validation_smoke.freeze.json'
    if core.verified(marker):return
    base.gate(ds,1);base.gate(ds,2)
    train={x.sample_id:x for x in core.manifest(ds,'train')};val=core.manifest(ds,'validation')
    labels=sorted({x.label for x in train.values()})
    desc={x['query_id']:x for x in core.read(cv.ROOT/ds/'descriptions.json')}
    files=[ROOT/'run.freeze.json'];empty=[]
    for shot in (1,2):
        vp=core.read(cal.ROOT/f'{shot}shot'/ds/'validation_plan.json')
        cp=core.read(cv.ROOT/ds/'validation_plan.json') if shot==2 else None
        conversations=[]
        for i,q in enumerate(val[:3]):
            method='circles+matched_decision' if shot==2 and i==1 else 'rices+matched_decision'
            plan=cp if method.startswith('circles') else vp
            ids=plan['methods'][method+'.full'][i]
            for sample in [q]+[train[j] for j in ids]:assert core.sha(sample.image_path)==sample.image_sha256
            conversations.append(base.conv(ds,q,[train[j] for j in ids],labels,method,desc[q.sample_id]))
        outputs=[]
        for name,part in [('pair',conversations[:2]),('repeat',conversations[:2]),('tail',conversations[2:])]:
            base.held();value=backend.forward(part,512)
            path=out/f'validation_{shot}.{name}.json';core.write(path,value);files.append(path);outputs.append(value)
        assert outputs[0]['raw']==outputs[1]['raw'],'Validation batch repeat drift'
        for value in outputs:
            assert len(value['texts'])==len(value['parsed'])==len(value['raw'])
            for text,parsed in zip(value['texts'],value['parsed']):
                assert isinstance(text,str)
                if not text.strip():
                    assert type(parsed) is dict and parsed.get('role')=='assistant'
                    empty.append({'shot':shot,'response':'official assistant parse with empty text; exact-match INVALID'})
    core.freeze(marker,files+base.files_for_cal(ds,1)+base.files_for_cal(ds,2))
    core.write(out/'validation_empty_response_observations.json',{'policy':POLICY,'observations':empty,'test_access':False})


def audit_results_v7(ds,shot,root):
    BASE_AUDIT_RESULTS(ds,shot,root)
    audit_path=Path(root)/f'{shot}shot'/ds/'independent_audit.json'
    audit=core.read(audit_path)
    audit['target_response_policy']=POLICY
    audit['runner_sha256']=core.sha(SCRIPT)
    audit['test_used_for_tuning']=False
    core.write(audit_path,audit)


def cpu_check_v7():
    manifest,import_freeze=setup()
    checks=backend_v4.self_check()
    assert base.prediction('', ['cat','dog'])==base.INVALID
    assert base.prediction(' Cat ', ['cat','dog'])=='cat'
    assert base.prediction('- cat', ['cat','dog'])==base.INVALID
    base.cpu_check()
    path=ROOT/'cpu_check.json';receipt=core.read(path)
    assert receipt['status']=='passed' and receipt['script_sha256']==core.sha(SCRIPT)
    receipt.update(v6_policy_checks=checks,imported_blocks_freeze_sha256=core.sha(import_freeze),
        imported_method_results=sum(v['method_count'] for v in manifest['blocks'].values()),
        exact_match_empty_is_invalid=True,other_unknown_schemas_rejected=True,test_access=False)
    core.write(path,receipt)
    assert core.verified(ROOT/'run.freeze.json')
    code_files=[SCRIPT,Path(backend_v4.__file__),Path(frozen.__file__),Path(frozen.audited.__file__),Path(base.__file__),Path(frozen.replay.__file__)]
    core.write(ROOT/'code_audit.json',{'status':'passed','runner_version':'v7_role_only_invalid_20260928',
        'files':{str(p.resolve()):core.sha(p) for p in code_files},
        'run_freeze_sha256':core.sha(ROOT/'run.freeze.json'),
        'imported_blocks_freeze_sha256':core.sha(import_freeze),
        'cpu_contract':checks,'test_access':False,'v5_immutable':True})
    print('Gemma v7 CPU contract checks passed; no GPU or test inference run',flush=True)


def run_all():
    manifest,import_freeze=setup()
    receipt=core.read(ROOT/'cpu_check.json')
    assert receipt.get('status')=='passed' and receipt.get('script_sha256')==core.sha(SCRIPT)
    assert receipt.get('imported_blocks_freeze_sha256')==core.sha(import_freeze)
    assert core.verified(ROOT/'run.freeze.json') and core.verified(import_freeze)
    assert core.read(ROOT/'code_audit.json').get('status')=='passed'
    for ds in DATASETS:
        base.gate(ds,1);base.gate(ds,2)
    import msvcrt
    handles=[]
    for path in (core.OLD/'gpu_queue.lock',core.QUEUE/'gpu_queue.lock'):
        f=path.open('a+b');f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);handles.append(f)
    base.emit('v6_target_started',dataset='dtd',audited_imported_method_results=12,test_used_for_tuning=False)
    for ds in DATASETS:
        if ds=='dtd':
            base.emit('target_block_imported_audited',dataset=ds,shots=[1,2],methods=12,
                source_root=str(V5_ROOT),source_manifest_sha256=core.sha(ROOT/'imported_blocks.json'),gpu_inference=False)
            continue
        backend=cv.Gemma12Backend()
        try:base.validate_prompt(ds,backend)
        finally:del backend;core.cleanup()
        base.inputs(ds);base.evidence(ds);base.plan(ds,1);base.audit_plan(ds,1,ROOT)
        backend=cv.Gemma12Backend()
        try:base.infer(ds,1,backend);base.test_descriptions(ds,backend)
        finally:del backend;core.cleanup()
        base.caption_embeddings(ds);base.plan(ds,2);base.audit_plan(ds,2,ROOT)
        backend=cv.Gemma12Backend()
        try:base.infer(ds,2,backend)
        finally:del backend;core.cleanup()
    base.emit('all_target_blocks_finished',completed_imported_methods=12,test_used_for_tuning=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['cpu-check','all']);args=ap.parse_args()
    setup()
    if args.stage=='cpu-check':cpu_check_v7();return
    run_all()

if __name__=='__main__':
    try:main()
    except Exception as exc:
        try:
            ROOT.mkdir(parents=True,exist_ok=True)
            core.write(ROOT/'DIAGNOSTIC_HOLD.json',{'version':'v7_role_only_invalid_20260928','error_type':type(exc).__name__,'error':str(exc),'time':time.time(),'test_used_for_tuning':False})
            base.emit('failed',error_type=type(exc).__name__,error=str(exc))
        except Exception:pass
        raise
