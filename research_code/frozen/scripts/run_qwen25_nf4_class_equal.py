"""Sequential four-dataset Qwen2.5 VL 7B NF4 discovery -> raw SigLIP -> class-equal ICL."""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from qwen25_nf4_backend import CONTRACT, MODEL, MODEL_ID, REVISION, Qwen25Backend
import run_cub_raw_main_methods as r
from dgdr.atoms import collapse_exact_duplicate_groups, filter_bank, group_semantic_text, lint_groups as schema_lint_groups, response_aware_deduplicate
from dgdr.audit import distribution_audit
from dgdr.cli import _balanced_discovery_subset
from dgdr.contracts import assert_discovery_is_train_only
from dgdr.discovery import make_discovery_requests, parse_discovery_response
from dgdr.encoders import SiglipEncoder
from dgdr.evidence import make_response_fingerprints
from dgdr.hashing import stable_hash
from dgdr.io import append_jsonl, read_jsonl, save_atom_bank
from dgdr.schema import AtomBank
from raw_siglip_evidence import build_raw_siglip_evidence
from run_qwen3_dataset_plan import paired

OUT = r.PROJECT / "artifacts/qwen25_nf4_class_equal_20260907"
PROTOCOL = r.PROJECT / "docs/qwen25_nf4_class_equal_20260907_zh.md"
ORDER = ("cub", "dtd", "aircraft", "dogs")
COUNTS = {"cub": (5993,1000,4794), "dtd": (1874,1872,1880),
          "dogs": (10726,1197,8580), "aircraft": (3334,3333,3333)}
PARENTS = {"rices_class_equal_alpha_raw": "rices", "atom_label_equal_alpha_raw": "muier"}
METHODS = ("random", "rices", "muier", *PARENTS)
SIGLIP = MODEL.parent / "siglip2-base-patch16-224"
MAX_STATES = 10


def lint_groups(groups):
    return schema_lint_groups(groups, max_states=MAX_STATES)


def verify_hash_tree(path):
    """Verify receipts without mistaking a vocabulary's 'files' token for one."""
    files = r.read_json(path)["files"]
    if not isinstance(files, dict):
        raise ValueError("invalid hash receipt: " + str(path))
    for filename, digest in files.items():
        if r.file_sha256(filename) != digest:
            raise ValueError("frozen file drift: " + filename)
        if Path(filename).suffix == ".json":
            value = r.read_json(filename)
            if isinstance(value, dict) and isinstance(value.get("files"), dict):
                verify_hash_tree(filename)


# Scope the verifier correction to this process; do not modify shared frozen sources.
r.verify = verify_hash_tree



def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("w",encoding="utf-8") as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary,path)


def append_context(path,row):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as handle:
        handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
        handle.flush()
        os.fsync(handle.fileno())


def held():
    if any((OUT / n).exists() for n in ("PAUSED_BY_USER.json", "DIAGNOSTIC_HOLD.json")):
        raise RuntimeError("Qwen2.5 NF4 run held; explicit intervention required")


def emit(stage, **kw):
    row = dict(stage=stage, time=time.time(), **kw)
    write_json(OUT / "status.json", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def data_dir(ds):
    return r.PROJECT / ("artifacts/cub_full_current_atoms_20260904" if ds == "cub" else
                        "artifacts/four_dataset_atom_label_20260905/" + ds)


def manifest_path(ds, split):
    return data_dir(ds) / f"{ds}.{split}.jsonl"


def embedding_path(ds, split):
    return data_dir(ds) / (("siglip." if ds == "cub" else "") + f"{split}.embeddings")


def load(ds, split):
    rows = r.load_manifest(manifest_path(ds, split), expected_split=split)
    emb = r.EmbeddingStore.load(embedding_path(ds, split))
    if len(rows) != COUNTS[ds][("train","validation","test").index(split)] or tuple(s.sample_id for s in rows) != emb.sample_ids:
        raise ValueError("original manifest count/order changed")
    return rows, emb


def instruction(ds):
    return ("Identify the bird species shown in the image." if ds == "cub" else
            r.read_json(data_dir(ds) / "dataset_config.json")["instruction"])


def frozen(path, files):
    if path.exists():
        r.verify(path)
    else:
        r.freeze(path, files)


def image_check(rows):
    for sample in rows:
        if r.file_sha256(sample.image_path) != sample.image_sha256:
            raise ValueError("image hash mismatch: " + sample.sample_id)


def initialize():
    held()
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / "run.freeze.json"
    if fp.exists():
        r.verify(fp)
        return
    smoke = r.read_json(OUT / "smoke.json")
    if smoke["status"] != "passed" or smoke["contract"] != CONTRACT:
        raise ValueError("new exact backend must pass smoke first")
    sources = [Path(__file__), Path(__file__).with_name("qwen25_nf4_backend.py"), PROTOCOL,
        Path(__file__).with_name("raw_siglip_evidence.py"), Path(__file__).with_name("run_cub_atom_label_residual.py"),
        Path(__file__).with_name("run_cub_full_current_atoms.py"), Path(__file__).with_name("run_cub_raw_main_methods.py"),
        Path(__file__).with_name("run_fgclip2_evidence_comparison.py"), Path(__file__).with_name("run_qwen3_dataset_plan.py")]
    sources += [Path(__file__).with_name("smoke_qwen25_nf4_class_equal.py"),
                r.PROJECT / "tests/test_qwen25_nf4_class_equal.py"]
    sources += sorted((r.PROJECT / "src/dgdr").rglob("*.py"))
    sources += sorted(p for p in MODEL.iterdir() if p.is_file())
    for ds in ORDER:
        out = OUT / ds
        rows, emb = load(ds, "train")
        val, _ = load(ds, "validation")
        image_check(rows + val)
        if set(s.image_sha256 for s in rows) & set(s.image_sha256 for s in val):
            raise ValueError("train/validation image overlap")
        labels = sorted({s.label for s in rows})
        selected = _balanced_discovery_subset(rows, 12, 42)
        requests = make_discovery_requests(sample_ids=[s.sample_id for s in selected],
            image_paths=[s.image_path for s in selected], task_instruction=instruction(ds),
            class_names=labels, batch_size=12, max_groups_per_perspective=6)
        write_json(out / "discovery_inputs.json", dict(labels=labels, selected=[s.to_dict() for s in selected],
            requests=[dict(prompt_id=q.prompt_id, batch_id=q.batch_id, sample_ids=list(q.sample_ids),
                image_paths=list(q.image_paths), perspective=q.perspective, prompt=q.prompt) for q in requests]))
        sources += [out / "discovery_inputs.json"]
        for split in ("train","validation"):
            sources += [manifest_path(ds,split)] + r.paired_files(embedding_path(ds,split))
    write_json(OUT / "run_contract.json", dict(target=CONTRACT, datasets=COUNTS, methods=METHODS,
        evidence=r.POLICY, new_target_discovery=True, inherited_weights=False,
        test_used_for_selection=False, discovery_budget=12, images_per_request=12, perspectives=6,
        max_groups_per_request=6, max_accepted_states_per_group=MAX_STATES,
        elicitation_states="2--6 requested; up to 10 accepted by user amendment",
        malformed_response_retries=1, response_dedup=dict(semantic=.90,response=.98,pairs=4096),
        alpha_grid=r.base.ALPHAS.tolist(), folds=5, seed=42, dataset_execution_order=ORDER,
        per_dataset_complete_before_next=True,
        packages={p:importlib.metadata.version(p) for p in ("torch","transformers","numpy","scipy","pillow","accelerate","bitsandbytes")}))
    frozen(fp, sources + [OUT / "smoke.json", OUT / "run_contract.json"])
    emit("initialized", test_used=False)


def discover(ds, backend):
    held()
    out = OUT / ds
    if (out / "discovery.complete.json").exists():
        r.verify(out / "discovery.complete.json")
        return
    from dgdr.discovery import DiscoveryRequest
    inp = r.read_json(out / "discovery_inputs.json")
    groups, rejected = [], []
    for index, value in enumerate(inp["requests"]):
        request = DiscoveryRequest(**{**value, "sample_ids":tuple(value["sample_ids"]), "image_paths":tuple(value["image_paths"])})
        success = False
        for attempt in range(2):
            held()
            path = out / f"discovery_response_{index:02d}_{attempt}.json"
            prompt = request.prompt + ("\nYour previous response was not valid for the required schema. Return a complete valid JSON object only." if attempt else "")
            if path.exists():
                response = r.read_json(path)
                if response["request_hash"] != stable_hash([value, CONTRACT, prompt]):
                    raise ValueError("discovery response provenance changed")
                raw = response["raw"]
            else:
                emit("discovering", dataset=ds, perspective=index+1, total=6, attempt=attempt)
                start = time.time()
                raw = backend.generate(request.image_paths, prompt)
                write_json(path, dict(request_hash=stable_hash([value,CONTRACT,prompt]), raw=raw,
                    source_model=backend.model_id, sample_ids=request.sample_ids, seconds=time.time()-start,
                    generation_stats=getattr(backend,"last_generation_stats",{})))
            try:
                parsed = parse_discovery_response(raw, request=request, source_model=backend.model_id)
                if not parsed or len(parsed)>6 or any(i.level=="error" for i in lint_groups(parsed)):
                    raise ValueError("empty/oversized/schema-invalid response")
                groups.extend(parsed)
                success = True
                break
            except (ValueError,TypeError) as error:
                rejected.append(dict(perspective=index,attempt=attempt,error=str(error)))
        if not success:
            write_json(out / "discovery_rejections.json", rejected)
            raise RuntimeError("discovery schema failure after fixed retry; do not silently omit perspective")
    bank = AtomBank(task_name=ds,task_instruction=instruction(ds),class_names=tuple(inp["labels"]),
        factor_scope="task",groups=tuple(groups),discovery_split="train",discovery_seed=42)
    assert_discovery_is_train_only(bank,load(ds,"train")[0])
    save_atom_bank(out / "atom_bank.raw.json",bank)
    keep,removed = collapse_exact_duplicate_groups(bank.groups)
    exact = filter_bank(bank,keep)
    save_atom_bank(out / "atom_bank.exact.json",exact)
    write_json(out / "exact_dedup.json",dict(removed_to_representative=removed,lint=[i.to_dict() for i in lint_groups(exact.groups)]))
    write_json(out / "discovery_rejections.json",rejected)
    frozen(out / "discovery.complete.json",[OUT / "run.freeze.json",out / "atom_bank.raw.json",out / "atom_bank.exact.json",
        out / "exact_dedup.json",out / "discovery_rejections.json"]+sorted(out.glob("discovery_response_*.json")))
    emit("discovered",dataset=ds,groups=len(exact.groups),states=sum(len(g.states) for g in exact.groups))


def evidence(ds,split,bank,encoder,name=None):
    rows,emb = load(ds,split)
    prefix = OUT / ds / (name or f"siglip.{split}.evidence")
    result = build_raw_siglip_evidence(samples=rows,image_embeddings=emb,bank=bank,encoder=encoder)
    r.validate_alignment(rows,emb,result,bank)
    result.save(prefix)
    write_json(Path(str(prefix)+".distribution.json"),distribution_audit(bank,result))
    return rows,emb,result


def calibrate(ds,encoder):
    held()
    out = OUT / ds
    if (out / "calibration.freeze.json").exists():
        r.verify(out / "calibration.freeze.json")
        return
    r.verify(out / "discovery.complete.json")
    bank = r.load_atom_bank(out / "atom_bank.exact.json")
    train,te,raw = evidence(ds,"train",bank,encoder,"exact.train.evidence")
    dedup = response_aware_deduplicate(group_ids=list(raw.group_ids),
        semantic_embeddings=encoder.encode_texts([group_semantic_text(g) for g in bank.groups]),
        response_fingerprints=make_response_fingerprints(raw,pair_count=4096,seed=42),
        semantic_threshold=.90,response_threshold=.98)
    bank = filter_bank(bank,dedup.kept_group_ids)
    save_atom_bank(out / "atom_bank.final.json",bank)
    write_json(out / "response_dedup.json",dedup.to_dict())
    train,te,tv = evidence(ds,"train",bank,encoder)
    validation,ve,vv = evidence(ds,"validation",bank,encoder)
    labels,_,ty,vy = r.shared.label_contract(train,validation)
    text = encoder.encode_texts([f"a photo of a {label}" for label in labels])
    np.save(out / "label_embeddings.npy",text)
    write_json(out / "label_order.json",list(labels))
    prototypes = r.build_class_atom_prototypes(samples=train,evidence=tv)
    assert prototypes.labels == labels
    folds = r.stratified_folds(validation,5,42)
    methods = {}
    for method,parent in PARENTS.items():
        f,_ = r.shared.class_features(ve,vv,te,ty,text if parent=="muier" else np.zeros_like(text),prototypes)
        oof,history = np.empty(len(vy),np.int64),[]
        for fold in range(5):
            fit,hold = np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
            alpha,score = r.shared.select_alpha(f,vy,fit)
            weights = r.shared.equal_weights(f.shape[2],alpha)
            oof[hold] = r.score_classes(f[hold],weights).argmax(1)
            history.append(dict(fold=fold,fit_count=len(fit),heldout_count=len(hold),alpha=alpha,weights=weights.tolist(),fit_same_class=score))
        alpha,score = r.shared.select_alpha(f,vy,np.arange(len(vy)))
        oa,ba = float(np.mean(oof==vy)),float(np.mean(f[:,:,0].argmax(1)==vy))
        methods[method] = dict(parent=parent,alpha=alpha,weights=r.shared.equal_weights(f.shape[2],alpha).tolist(),
            full_fit_same_class=score,outer_folds=history,oof_same_class=oa,baseline_same_class=ba,
            selected_deployment=method if oa>ba+1e-12 else parent,independent_fit=True)
        emit("calibrated",dataset=ds,method=method,alpha=alpha,oof=oa,baseline=ba)
    write_json(out / "calibration.json",dict(methods=methods,bank_hash=bank.bank_hash,folds=folds.tolist(),
        inherited_weights=False,test_used=False,supervision="same-class demo, not target accuracy"))
    files=[out / "discovery.complete.json",out / "atom_bank.final.json",out / "response_dedup.json",
        out / "calibration.json",out / "label_embeddings.npy",out / "label_order.json"]
    for name in ("exact.train.evidence","siglip.train.evidence","siglip.validation.evidence"):
        files += r.paired_files(out/name)+[out/(name+".distribution.json")]
    frozen(out / "calibration.freeze.json",files)


def prepare_test(ds,encoder):
    held()
    out=OUT/ds
    if (out/"test.freeze.json").exists():
        r.verify(out/"test.freeze.json")
        return
    r.verify(out/"calibration.freeze.json")
    bank=r.load_atom_bank(out/"atom_bank.final.json")
    train,te=load(ds,"train")
    tv=r.EvidenceStore.load(out/"siglip.train.evidence")
    query,qe,qv=evidence(ds,"test",bank,encoder)
    image_check(query)
    development=train+load(ds,"validation")[0]
    if set(s.image_sha256 for s in query)&set(s.image_sha256 for s in development):
        raise ValueError("development/test exact-image overlap")
    labels,_,ty,_=r.shared.label_contract(train,[])
    text=np.load(out/"label_embeddings.npy")
    if list(labels)!=r.read_json(out/"label_order.json"):
        raise ValueError("class order mismatch")
    prototypes=r.build_class_atom_prototypes(samples=train,evidence=tv)
    cal=r.read_json(out/"calibration.json")
    tids=np.asarray(te.sample_ids)
    methods={m:[] for m in METHODS}
    old=r.read_json(r.PROJECT/"artifacts/five_method_mainline_20260906"/ds/"resolved_demo_plan.json")
    if old["query_ids"]!=list(qe.sample_ids):
        raise ValueError("reference query order changed")
    methods["random"]=old["methods"]["random"]
    for start in range(0,len(query),64):
        stop=min(start+64,len(query))
        eb=r.EmbeddingStore(qe.sample_ids[start:stop],qe.embeddings[start:stop],qe.model_id,qe.model_revision)
        ev=r.shared.sliced_evidence(qv,start,stop)
        for method,parent in PARENTS.items():
            lt=text if parent=="muier" else np.zeros_like(text)
            f,best=r.shared.class_features(eb,ev,te,ty,lt,prototypes)
            base=r.shared.classes_to_demo_ids(f[:,:,0].argmax(1),best,tids)
            direct=eb.embeddings@te.embeddings.T+(eb.embeddings@lt.T)[:,ty]
            if base!=tids[direct.argmax(1)].tolist():
                raise ValueError("not complete-pool retrieval")
            methods[parent].extend(base)
            methods[method].extend(r.shared.classes_to_demo_ids(r.score_classes(f,np.asarray(cal["methods"][method]["weights"],np.float32)).argmax(1),best,tids))
    for parent in ("rices","muier"):
        if methods[parent]!=old["methods"][parent]:
            raise ValueError("unchanged baseline parity failed: "+parent)
    for demos in methods.values():
        if len(demos)!=len(query) or not set(demos)<=set(tids):
            raise ValueError("incomplete plan/nontrain demo")
    write_json(out/"test_plan.json",dict(query_ids=list(qe.sample_ids),methods=methods,shots=1,
        candidate_pool="complete train",test_labels_used=False,selected_deployment={m:p["selected_deployment"] for m,p in cal["methods"].items()}))
    frozen(out/"test.freeze.json",[out/"calibration.freeze.json",out/"test_plan.json",manifest_path(ds,"test"),
        r.PROJECT/"artifacts/five_method_mainline_20260906"/ds/"resolved_demo_plan.json"]+
        r.paired_files(embedding_path(ds,"test"))+r.paired_files(out/"siglip.test.evidence"))
    unique=sum(len({methods[m][i] for m in METHODS}) for i in range(len(query)))
    write_json(out/"workload.json",dict(queries=len(query),unique_contexts=unique,method_cells=len(query)*len(METHODS)))
    emit("test_plan_frozen",dataset=ds,queries=len(query),methods=len(methods),unique_contexts=unique)


def evaluate(ds,backend):
    held()
    out=OUT/ds
    r.verify(out/"test.freeze.json")
    if (out/"complete.json").exists():
        r.verify(out/"complete.json")
        return
    train={s.sample_id:s for s in load(ds,"train")[0]}
    query={s.sample_id:s for s in load(ds,"test")[0]}
    image_check(list(train.values())+list(query.values()))
    labels=sorted({s.label for s in train.values()})
    codes,ids=backend.class_codes(len(labels))
    plan=r.read_json(out/"test_plan.json")
    protocol=dict(contract=CONTRACT,labels=labels,codes=codes,token_ids=ids,instruction=instruction(ds),
        plan_hash=r.file_sha256(out/"test_plan.json"),run_hash=r.file_sha256(OUT/"run.freeze.json"))
    ph=stable_hash(protocol)
    pp=out/"inference_protocol.json"
    if pp.exists() and r.read_json(pp)!=protocol:
        raise ValueError("target protocol changed")
    write_json(pp,protocol)
    cache_path=out/"context_cache.jsonl"
    cache={}
    if cache_path.exists():
        for row in read_jsonl(cache_path):
            if row["protocol_hash"]!=ph:
                raise ValueError("foreign-model context cache")
            if row["key"] in cache and cache[row["key"]]!=row:
                raise ValueError("conflicting context cache")
            cache[row["key"]]=row
    truth={sid:s.label for sid,s in query.items()}
    predictions,results={},{}
    for method in METHODS:
        pred={}
        for index,(qid,did) in enumerate(zip(plan["query_ids"],plan["methods"][method])):
            held()
            q,d=query[qid],train[did]
            key=stable_hash([ph,qid,did,q.image_sha256,d.image_sha256])
            if key not in cache:
                message=backend.conversation(instruction(ds),q.image_path,d.image_path,d.label,labels,codes)
                selected,logits,stats=backend.classify(message,ids)
                row=dict(key=key,protocol_hash=ph,query_id=qid,demo_id=did,query_image_sha256=q.image_sha256,
                    demo_image_sha256=d.image_sha256,prediction=labels[selected],logits=logits,stats=stats)
                append_context(cache_path,row)
                cache[key]=row
            pred[qid]=cache[key]["prediction"]
            if (index+1)%50==0:
                emit("target_running",dataset=ds,method=method,done=index+1,total=len(query),unique_contexts=len(cache))
        predictions[method]=pred
        acc=float(np.mean([pred[q]==truth[q] for q in truth]))
        macro=float(np.mean([np.mean([pred[q]==label for q in truth if truth[q]==label]) for label in sorted(set(truth.values()))]))
        results[method]=dict(accuracy=acc,macro_accuracy=macro,n=len(query))
        if method in PARENTS:
            results[method]["paired_vs_parent"]=paired(predictions[PARENTS[method]],pred,truth)
        write_json(out/"predictions.json",predictions)
        write_json(out/"results.json",dict(dataset=ds,complete_methods=list(results),methods=results,
            p_values="raw exact McNemar; unadjusted exploratory",test_used_for_tuning=False))
        emit("method_complete",dataset=ds,method=method,**results[method])
    frozen(out/"complete.json",[out/"test.freeze.json",pp,cache_path,out/"predictions.json",out/"results.json"])
    emit("dataset_complete",dataset=ds)



def audit_report(ds):
    from dgdr.metrics import accuracy_metrics
    out=OUT/ds
    r.verify(out/"complete.json")
    pred=r.read_json(out/"predictions.json")
    result=r.read_json(out/"results.json")
    queries=load(ds,"test")[0]
    truth={s.sample_id:s.label for s in queries}
    if set(pred)!=set(METHODS) or set(result["methods"])!=set(METHODS):
        raise ValueError("incomplete five-method result")
    for method in METHODS:
        if set(pred[method])!=set(truth):
            raise ValueError("test prediction coverage mismatch")
        metrics=accuracy_metrics(list(truth.values()),[pred[method][q] for q in truth])
        for k,v in metrics.items():
            if abs(result["methods"][method][k]-v)>1e-10:
                raise ValueError("metric recomputation failed")
        if method in PARENTS:
            stats=paired(pred[PARENTS[method]],pred[method],truth)
            if stats!=result["methods"][method]["paired_vs_parent"]:
                raise ValueError("paired recomputation failed")
    lines=["# Qwen2.5 VL 7B NF4 — "+ds,"","Full test; 1-shot; raw SigLIP evidence; independently calibrated class-equal methods.",
        "","| Method | Accuracy | Macro |","|---|---:|---:|"]
    for method in METHODS:
        v=result["methods"][method]
        lines.append(f"| {method} | {v['accuracy']*100:.3f}% | {v['macro_accuracy']*100:.3f}% |")
    lines+=["","Unadjusted, exploratory paired comparisons; no test tuning."]
    for method in PARENTS:
        s=result["methods"][method]["paired_vs_parent"]
        lines.append(f"- {method}: rescue/harm {s['rescue']}/{s['harm']}; delta {s['accuracy_delta']*100:.3f} pp; raw p={s['exact_mcnemar_p']:.6g}; CI={s['paired_bootstrap_95pct']}.")
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(out/"numeric_audit.json",dict(status="passed",methods=len(METHODS),queries=len(queries)))
    frozen(out/"audit.complete.json",[out/"complete.json",out/"report.md",out/"numeric_audit.json"])
    emit("dataset_audited",dataset=ds)


def assert_previous_complete(ds):
    for previous in ORDER[:ORDER.index(ds)]:
        marker=OUT/previous/"audit.complete.json"
        if not marker.exists():
            raise RuntimeError("previous dataset must finish first: "+previous)
        r.verify(marker)


def run_dataset(ds):
    import subprocess
    assert_previous_complete(ds)
    out=OUT/ds
    stages=(("discovery","discovery.complete.json"),("prepare","test.freeze.json"),("target","audit.complete.json"))
    for stage,marker in stages:
        held()
        path=out/marker
        if path.exists():
            r.verify(path)
            continue
        stamp=time.strftime("%Y%m%d_%H%M%S")
        stdout=out/f"{stage}_{stamp}.stdout.log"
        stderr=out/f"{stage}_{stamp}.stderr.log"
        env={**os.environ,"PYTHONUTF8":"1","PYTHONIOENCODING":"utf-8"}
        emit("stage_starting",dataset=ds,worker=stage,stdout=str(stdout),stderr=str(stderr))
        with stdout.open("w",encoding="utf-8") as fo,stderr.open("w",encoding="utf-8") as fe:
            subprocess.run([sys.executable,"-u",str(Path(__file__).resolve()),"--worker",stage,"--dataset",ds],
                cwd=r.PROJECT.parent,env=env,stdout=fo,stderr=fe,check=True)
        if not path.exists():
            raise RuntimeError("worker returned without checkpoint: "+str(path))
        r.verify(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--worker",choices=("discovery","prepare","target"))
    parser.add_argument("--dataset",choices=ORDER)
    parser.add_argument("--init-only",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.worker:
        if not args.dataset:
            parser.error("--worker requires --dataset")
        held()
        r.verify(OUT/"run.freeze.json")
        assert_previous_complete(args.dataset)
        if args.worker=="discovery":
            discover(args.dataset,Qwen25Backend())
        elif args.worker=="prepare":
            encoder=SiglipEncoder(str(SIGLIP),batch_size=64,dtype="float16")
            calibrate(args.dataset,encoder)
            prepare_test(args.dataset,encoder)
        else:
            evaluate(args.dataset,Qwen25Backend())
            audit_report(args.dataset)
        return
    initialize()
    if args.init_only:
        return
    for ds in ORDER:
        run_dataset(ds)
    summary={ds:r.read_json(OUT/ds/"results.json") for ds in ORDER}
    write_json(OUT/"summary.json",summary)
    frozen(OUT/"complete.json",[OUT/d/"audit.complete.json" for d in ORDER]+[OUT/"summary.json"])
    emit("all_complete",datasets=len(ORDER),methods=len(METHODS))


if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        emit("failed",error_type=type(exc).__name__,error=str(exc))
        raise
