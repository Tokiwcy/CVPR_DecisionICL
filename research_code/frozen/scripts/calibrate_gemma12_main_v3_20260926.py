"""Gemma-specific train/validation calibration for RICES and MUIER, no test IO.
CIRCLES calibration is a separate dependency and is NOT completed by this script.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import prepare_gemma12_decision_bank_v3_20260926 as prep
from gemma12_nf4_backend_v2_20260926 import MODEL_ID
from dgdr.atom_label import build_class_atom_prototypes, score_classes
from dgdr.calibration import stratified_folds

core = prep.core
SCRIPT = Path(__file__).resolve()
ROOT = prep.ROOT.parent / 'calibration'
CONTRACT = dict(target=prep.CONTRACT, source='matched_decision',
    objective='validation class accuracy', folds=5, seed=42,
    class_selection='class aggregate, argmax class, descending parent within class',
    alpha_grid=list(core.common.ALPHAS), alpha_tie='more parent mass',
    independently_refit_each_dataset_shot_parent=True, test_access=False,
    scope='RICES and MUIER only; CIRCLES pending separately')

def held():
    for root in (ROOT, prep.ROOT, prep.ROOT.parent, core.QUEUE):
        if any((root/n).exists() for n in ('PAUSED_BY_USER.json','DIAGNOSTIC_HOLD.json','WATCHDOG_STOP')):
            raise RuntimeError('Gemma calibration held')

def dependencies():
    project = SCRIPT.parents[1]
    return sorted({Path(m.__file__).resolve() for m in list(sys.modules.values())
        if getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py'
        and Path(m.__file__).resolve().is_relative_to(project)})

def choose(scores, train_y, classes, shots):
    picks=[]
    for row, cls in enumerate(classes):
        candidates=np.flatnonzero(train_y==cls)
        if len(candidates)<shots:raise ValueError('Insufficient within-class demonstrations')
        # Same frozen tie order as the current official-prompt retrieval helpers.
        ordered=core.r.top(scores[row:row+1,candidates],shots)[0]
        picks.append(candidates[ordered].tolist())
    return picks

def independent_fit(features, truth, validation):
    folds=stratified_folds(validation,5,42)
    oof=np.empty(len(truth),np.int64);history=[]
    for fold in range(5):
        fit=np.flatnonzero(folds!=fold);hold=np.flatnonzero(folds==fold)
        alpha,score=core.common.fit_alpha(features,truth,fit)
        weights=core.common.cub.shared.equal_weights(features.shape[2],alpha)
        oof[hold]=score_classes(features[hold],weights).argmax(1)
        history.append(dict(fold=fold,alpha=alpha,weights=weights.tolist(),
            fit_class_accuracy=score,fit_count=len(fit),heldout_count=len(hold)))
    alpha,score=core.common.fit_alpha(features,truth,np.arange(len(truth)))
    weights=core.common.cub.shared.equal_weights(features.shape[2],alpha)
    return dict(alpha=alpha,weights=weights.tolist(),outer_folds=history,
        full_fit_class_accuracy=score,oof_class_accuracy=float(np.mean(oof==truth)),
        independently_fit=True,inherited_weights=False),oof,folds

def clip_inputs(ds, split, rows):
    # Validate both historical source manifest bytes and NPZ/metadata, without
    # loading any test data, caption, bank, calibration, or target output.
    location=Path(core.read(core.OLD/'execution_locations.json')[ds+'/1'])
    old=core.read(location/'plan.freeze.json')['files']
    cp=core.resource(ds)/f'clip.{split}.npz'
    files=[core.common.source(ds).manifest(split),cp,Path(str(cp)+'.json')]
    for p in files:
        assert old.get(str(p.resolve()))==core.sha(p),f'Unverified CLIP input: {p}'
    return core.r.legacy.load_embeddings(cp,[x.sample_id for x in rows]),files

def calibrate(ds, shots):
    held();out=ROOT/f'{shots}shot'/ds;marker=out/'calibration.freeze.json'
    if core.verified(marker):return
    evidence_freeze=prep.ROOT/ds/'train_validation_evidence.freeze.json'
    assert core.verified(evidence_freeze),'Gemma train/validation evidence not complete'
    assert core.verified(prep.ROOT/'run.freeze.json')
    bank=core.load_atom_bank(prep.ROOT/ds/prep.MODE/'atom_bank.final.json')
    assert all(g.source_model==MODEL_ID for g in bank.groups)
    train,te=core.common.load(ds,'train');val,ve=core.common.load(ds,'validation')
    labels,ty,vy=core.common.labels_and_y(train,val)
    tv=core.load_ev(ds,prep.MODE,'train');vv=core.load_ev(ds,prep.MODE,'validation')
    proto=build_class_atom_prototypes(samples=train,evidence=tv)
    assert proto.labels==labels
    ti,train_files=clip_inputs(ds,'train',train);vi,val_files=clip_inputs(ds,'validation',val)
    label_path=prep.ROOT/ds/'label_embeddings.npy';lv=np.load(label_path)
    assert lv.shape==(len(labels),te.embeddings.shape[1])
    parents={'rices':vi@ti.T,'muier':core.common.parent_scores(ve,te,ty,lv)}
    methods={};validation_plan={};all_folds=None
    for parent,scores in parents.items():
        held()
        feature=(core.r.official.single_channel_atom_features(scores,ty,vv,proto,len(labels))
            if parent=='rices' else core.common.cub.shared.class_features(ve,vv,te,ty,lv,proto)[0])
        result,oof,folds=independent_fit(feature,vy,val)
        full_classes=score_classes(feature,np.asarray(result['weights'],np.float32)).argmax(1)
        parent_picks=core.r.top(scores,shots).tolist()
        oof_picks=choose(scores,ty,oof,shots);full_picks=choose(scores,ty,full_classes,shots)
        def precision(picks):return float(np.mean([[ty[j]==vy[i] for j in ids] for i,ids in enumerate(picks)]))
        result.update(validation_parent_demo_precision=precision(parent_picks),
            validation_oof_demo_precision=precision(oof_picks),bank_hash=bank.bank_hash,
            shots=shots,parent=parent,source_model=MODEL_ID)
        methods[parent+'+matched_decision']=result
        validation_plan[parent]=parent_picks
        validation_plan[parent+'+matched_decision.oof']=oof_picks
        validation_plan[parent+'+matched_decision.full']=full_picks
        all_folds=folds.tolist()
        del feature
    payload=dict(dataset=ds,shots=shots,contract=CONTRACT,methods=methods,folds=all_folds,
        train_ids=[x.sample_id for x in train],validation_ids=[x.sample_id for x in val],
        test_used=False,inherited_weights=False,bank_hash=bank.bank_hash)
    core.write(out/'calibration.json',payload)
    core.write(out/'validation_plan.json',dict(query_ids=payload['validation_ids'],
        methods={m:[[train[j].sample_id for j in ids] for ids in picks] for m,picks in validation_plan.items()},test_access=False))
    core.freeze(marker,dependencies()+[evidence_freeze,prep.ROOT/'run.freeze.json',
        out/'calibration.json',out/'validation_plan.json']+train_files+val_files)
    print(core.json.dumps(dict(stage='baseline_parent_calibration_complete',dataset=ds,shots=shots,
        methods=list(methods),test_access=False)),flush=True)

def cpu_check():
    scores=np.asarray([[9.,1.,7.,8.],[0.,4.,3.,2.]],np.float32)
    ty=np.asarray([0,0,1,1]);cls=np.asarray([1,0])
    assert choose(scores,ty,cls,1)==[[3],[1]]
    assert choose(scores,ty,cls,2)==[[3,2],[1,0]]
    for ds in core.ORDER:
        train=core.manifest(ds,'train');val=core.manifest(ds,'validation')
        _,ty,_=core.common.labels_and_y(train,val)
        assert min(np.bincount(ty))>=2
        folds=stratified_folds(val,5,42)
        assert set(folds)==set(range(5))
        assert not {x.image_sha256 for x in train}&{x.image_sha256 for x in val}
        clip_inputs(ds,'train',train);clip_inputs(ds,'validation',val)
    validation=core.manifest('dtd','validation')
    _,_,truth=core.common.labels_and_y(core.manifest('dtd','train'),validation)
    f=np.zeros((len(truth),int(truth.max())+1,2),np.float32)
    f[np.arange(len(truth)),(truth+1)%f.shape[1],0]=1
    f[np.arange(len(truth)),truth,1]=1
    atom_result,atom_oof,_=independent_fit(f,truth,validation)
    parent_result,parent_oof,_=independent_fit(f[:,:,::-1].copy(),truth,validation)
    assert np.all(atom_oof==truth) and np.all(parent_oof==truth)
    assert atom_result['alpha']!=parent_result['alpha'], 'Fit inherited another variant weights'
    assert len(atom_result['outer_folds'])==len(parent_result['outer_folds'])==5
    core.write(ROOT/'cpu_check.json',dict(status='passed',script_sha256=core.sha(SCRIPT),
        contract=CONTRACT,datasets=5,selection_checks=True,test_access=False,
        actual_gemma_calibration_started=False))
    print('CPU contract passed: five datasets, selection, folds, verified train/validation CLIP only',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['cpu-check','fit'])
    ap.add_argument('--dataset',choices=core.ORDER);ap.add_argument('--shots',type=int,choices=[1,2]);args=ap.parse_args()
    if args.stage=='cpu-check':cpu_check();return
    cpu=core.read(ROOT/'cpu_check.json');assert cpu['status']=='passed' and cpu['script_sha256']==core.sha(SCRIPT)
    for ds in ([args.dataset] if args.dataset else core.ORDER):
        for shots in ([args.shots] if args.shots else (1,2)):calibrate(ds,shots)

if __name__=='__main__':main()
