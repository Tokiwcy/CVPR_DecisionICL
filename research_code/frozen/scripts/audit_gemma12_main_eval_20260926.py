"""Read-only plan replay and independent statistics for Gemma main table."""
import numpy as np
from pathlib import Path
import time
import calibrate_gemma12_main_v3_20260926 as cal
import run_qwen3_circles_five as circles
from dgdr.metrics import accuracy_metrics,paired_comparison
core=cal.core

def audit_plan(ds,shot,root):
    root=Path(root);out=root/f'{shot}shot'/ds;res=root/'resources'/ds
    assert core.verified(out/'plan.freeze.json')
    rows=core.read(out/'test_plan.json');train,te=core.common.load(ds,'train');query,qe=core.common.load(ds,'test')
    assert [x['query_id'] for x in rows]==[x.sample_id for x in query]
    labels=tuple(sorted({x.label for x in train}));lookup={x:i for i,x in enumerate(labels)}
    ty=np.asarray([lookup[x.label] for x in train]);ids=[x.sample_id for x in train]
    ti=circles.load_embeddings(core.resource(ds)/'clip.train.npz',ids)
    qi=circles.load_embeddings(core.resource(ds)/'clip.test.npz',[x.sample_id for x in query])
    tv=core.load_ev(ds,cal.prep.MODE,'train');ev=core.common.EvidenceStore.load(res/'siglip.test.evidence')
    assert tv.bank_hash==ev.bank_hash and ev.sample_ids==qe.sample_ids
    proto=cal.build_class_atom_prototypes(samples=train,evidence=tv)
    lv=np.load(cal.prep.ROOT/ds/'label_embeddings.npy')
    fits=core.read(cal.ROOT/f'{shot}shot'/ds/'calibration.json')['methods']
    captions=None;desc={}
    if shot==2:
        fits.update(core.read(cal.prep.ROOT.parent/'circles_validation'/ds/'calibration.json')['methods'])
        captions=circles.load_embeddings(res/'clip.captions.test.npz',[x.sample_id for x in query])
        desc={x['query_id']:x for x in core.read(res/'descriptions.json')}
    rng=np.random.default_rng(42)
    methods=['random','rices','muier','rices+matched_decision','muier+matched_decision']+(['circles','circles+matched_decision'] if shot==2 else [])
    for begin in range(0,len(query),64):
        stop=min(begin+64,len(query));eb=core.common.EmbeddingStore(qe.sample_ids[begin:stop],qe.embeddings[begin:stop],qe.model_id,qe.model_revision)
        vb=core.common.cub.shared.sliced_evidence(ev,begin,stop)
        ps=qi[begin:stop]@ti.T;mp=core.common.parent_scores(eb,te,ty,lv)
        expected={'rices':np.argsort(ps,axis=1)[:,-shot:][:,::-1], 'muier':np.argsort(mp,axis=1)[:,-shot:][:,::-1]}
        for parent,scores in (('rices',ps),('muier',mp)):
            f=(core.r.official.single_channel_atom_features(ps,ty,vb,proto,len(labels)) if parent=='rices' else core.common.cub.shared.class_features(eb,vb,te,ty,lv,proto)[0])
            pred=(f@np.asarray(fits[parent+'+matched_decision']['weights'],np.float32)).argmax(1)
            picks=[]
            for i,c in enumerate(pred):
                candidates=np.flatnonzero(ty==c);picks.append(candidates[np.argsort(scores[i,candidates])[-shot:][::-1]])
            expected[parent+'+matched_decision']=picks
        if shot==2:
            cs=captions[begin:stop]@ti.T;f,oi,ci=circles.circle_atom_features(ps,cs,ty,vb,proto,len(labels))
            pred=(f@np.asarray(fits['circles+matched_decision']['weights'],np.float32)).argmax(1)
            expected['circles']=list(zip(circles.last_argmax(ps),circles.last_argmax(cs)))
            expected['circles+matched_decision']=[(oi[i,c],ci[i,c]) for i,c in enumerate(pred)]
        for i,row in enumerate(rows[begin:stop]):
            assert list(row['methods'])==methods or set(row['methods'])==set(methods)
            expected_random=[ids[int(j)] for j in rng.choice(len(ids),size=shot,replace=False)]
            assert row['methods']['random']==expected_random
            for m,pp in expected.items():assert row['methods'][m]==[ids[int(j)] for j in pp[i]],(ds,shot,begin+i,m)
            if shot==2:
                d=desc[row['query_id']];assert row['attribute']==d['attribute'] and row['modified_caption']==d['modified_caption']
    core.write(out/'independent_plan_audit.json',dict(status='passed',time=time.time(),dataset=ds,shots=shot,
        plan_freeze_sha256=core.sha(out/'plan.freeze.json'),plan_sha256=core.sha(out/'test_plan.json'),
        all_query_methods_replayed=True,test_labels_used=False,queries=len(query),methods=len(methods)))

def audit_results(ds,shot,root):
    root=Path(root);out=root/f'{shot}shot'/ds
    assert core.verified(out/'complete.json')
    truth={x.sample_id:x.label for x in core.manifest(ds,'test')}
    predictions=core.read(out/'predictions.json');saved=core.read(out/'results.json');checked={}
    for m,p in predictions.items():
        assert set(p)==set(truth)
        metrics=accuracy_metrics(list(truth.values()),[p[k] for k in truth])
        for key,value in metrics.items():assert abs(value-saved['methods'][m]['metrics'][key])<1e-12
        invalid=sum(v=='__invalid_exact_output__' for v in p.values())
        assert invalid==saved['methods'][m]['invalid_exact_output_count']
        checked[m]=dict(metrics=metrics,invalid=invalid)
        if '+' in m:
            pair=paired_comparison(labels=list(truth.values()),baseline_predictions=[predictions[m.split('+')[0]][k] for k in truth],method_predictions=[p[k] for k in truth])
            assert pair==saved['methods'][m]['paired_vs_parent']
            checked[m]['paired_vs_parent']=pair
    core.write(out/'independent_audit.json',dict(status='passed',time=time.time(),dataset=ds,shots=shot,
        complete_sha256=core.sha(out/'complete.json'),results_sha256=core.sha(out/'results.json'),
        predictions_sha256=core.sha(out/'predictions.json'),queries=len(truth),methods=checked))
