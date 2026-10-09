"""Gemma12 target-specific Decision bank and train/validation evidence only."""
import argparse,os,time
from pathlib import Path
import run_qwen3_decision_generic_20260920 as core
import gemma12_discovery_format_20260926 as format_compat
core.fmt=format_compat
from gemma12_nf4_backend_v2_20260926 import Gemma12Backend,MODEL,CONTRACT
SCRIPT=Path(__file__).resolve()
ROOT=core.QUEUE/'gemma12_main_1_2shot_v3_20260926/preparation'
DOC=core.r.PROJECT/'docs/gemma12_main_1_2shot_v3_20260926_zh.md'
PREFLIGHT=core.QUEUE/'gemma12_full_prompt_preflight_v3_20260926'
MODE='matched_decision'
core.ROOT=ROOT;core.MODES=(MODE,);core.DOC=DOC;core.SCRIPT=SCRIPT

def initialize():
    core.held();ROOT.mkdir(exist_ok=True,parents=True)
    if core.verified(ROOT/'run.freeze.json'):return
    files=[SCRIPT,DOC,Path(core.__file__),Path(format_compat.__file__),SCRIPT.with_name('gemma12_nf4_backend_v2_20260926.py'),
        SCRIPT.with_name('preflight_gemma4_12b_nf4_20260920.py'),
        SCRIPT.with_name('cars_caltech_atom_format.py'),SCRIPT.with_name('raw_siglip_evidence.py'),
        MODEL/'config.json',MODEL/'processor_config.json',MODEL/'tokenizer_config.json',MODEL/'chat_template.jinja']
    counts={}
    for ds in core.ORDER:
        split_rows={s:core.manifest(ds,s) for s in ('train','validation')}
        assert not {x.image_sha256 for x in split_rows['train']}.intersection(x.image_sha256 for x in split_rows['validation'])
        for split,rows in split_rows.items():
            assert len({x.sample_id for x in rows})==len(rows)
            loaded,embedding=core.common.load(ds,split)
            assert embedding.sample_ids==tuple(x.sample_id for x in rows)
            assert 'siglip2' in embedding.model_id.lower(), 'encoder contract mismatch'
            for x in rows:
                assert core.sha(x.image_path)==x.image_sha256, f'image bytes drift: {ds}/{x.sample_id}'
            files += [core.common.source(ds).manifest(split)]+core.pairfiles(core.common.source(ds).embedding(split))
        inputs=core.requests(ds,MODE)
        assert len(inputs['requests'])==18
        train_ids={x.sample_id for x in split_rows['train']}
        assert all(set(x['sample_ids']).issubset(train_ids) for x in inputs['requests'])
        for x in inputs['selected']:assert core.sha(x['image_path'])==x['image_sha256']
        path=ROOT/ds/MODE/'discovery_inputs.json';core.write(path,inputs);files.append(path)
        counts[ds]={s:len(v) for s,v in split_rows.items()}
    core.write(ROOT/'cpu_preflight.json',dict(status='passed',contract=CONTRACT,counts=counts,
        test_manifest_loaded=False,train_only_discovery=True,source=MODE,requests_per_dataset=18))
    core.freeze(ROOT/'run.freeze.json',files+[ROOT/'cpu_preflight.json'])
    core.emit('cpu_contract_passed',datasets=5,test_access=False)

def prepare(ds):
    marker=ROOT/ds/'train_validation_evidence.freeze.json'
    if core.verified(marker):return
    out=ROOT/ds/MODE
    assert core.verified(out/'discovery.complete.json')
    encoder=core.SiglipEncoder(str(core.SIGLIP),batch_size=64,dtype='float16')
    try:
        labels=sorted({x.label for x in core.manifest(ds,'train')})
        core.np.save(ROOT/ds/'label_embeddings.npy',encoder.encode_texts([f'a photo of a {x}' for x in labels]))
        bank=core.load_atom_bank(out/'atom_bank.exact.json')
        train,embedding=core.common.load(ds,'train')
        raw=core.build_raw_siglip_evidence(samples=train,image_embeddings=embedding,bank=bank,encoder=encoder)
        dedup=core.engine.response_aware_deduplicate(group_ids=list(raw.group_ids),
            semantic_embeddings=encoder.encode_texts([core.engine.group_semantic_text(g) for g in bank.groups]),
            response_fingerprints=core.engine.make_response_fingerprints(raw,pair_count=4096,seed=42),
            semantic_threshold=.90,response_threshold=.98)
        bank=core.filter_bank(bank,dedup.kept_group_ids);assert bank.groups
        core.save_atom_bank(out/'atom_bank.final.json',bank);core.write(out/'response_dedup.json',dedup.to_dict())
        files=[ROOT/'run.freeze.json',out/'discovery.complete.json',out/'atom_bank.final.json',out/'response_dedup.json',ROOT/ds/'label_embeddings.npy']
        for split in ('train','validation'):
            core.evidence(ds,MODE,split,bank,encoder);files+=core.pairfiles(out/f'siglip.{split}.evidence')
        core.freeze(marker,files)
        core.emit('train_validation_evidence_complete',dataset=ds,groups=len(bank.groups),test_access=False)
    finally:
        del encoder;core.cleanup()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['cpu-check','all']);args=ap.parse_args()
    initialize()
    if args.stage=='cpu-check':return
    assert core.verified(PREFLIGHT/'complete.json'),'full prompt preflight required'
    assert core.read(PREFLIGHT/'complete.json')['status']=='passed'
    assert core.read(PREFLIGHT/'complete.json')['contract']==CONTRACT
    import msvcrt
    handles=[]
    for p in (core.OLD/'gpu_queue.lock',core.QUEUE/'gpu_queue.lock'):
        f=p.open('a+b');f.seek(0)
        if not f.read(1):f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);handles.append(f)
    for ds in core.ORDER:
        core.held()
        if not core.verified(ROOT/ds/MODE/'discovery.complete.json'):
            backend=Gemma12Backend()
            try:core.discover(ds,backend)
            finally:del backend;core.cleanup()
        prepare(ds)
    core.emit('gemma12_train_validation_preparation_complete',datasets=5,test_access=False)

if __name__=='__main__':
    try:main()
    except Exception as error:
        core.emit('failed',error_type=type(error).__name__,error=str(error))
        core.write(ROOT/'DIAGNOSTIC_HOLD.json',dict(error=str(error),time=time.time()))
        raise
