"""Strict cache-to-batch provenance audit for Gemma; no model calls."""
import json
from pathlib import Path

def validate_cache(order,cache,batches,protocol,predict,labels,stable,sha,require_complete):
    assert set(cache).issubset(order),'Foreign cache context'
    if require_complete:assert set(cache)==set(order),'Incomplete cache coverage'
    expected_batches={str(i//2):order[i:i+2] for i in range(0,len(order),2)}
    assert set(batches).issubset(expected_batches),'Foreign raw batch'
    for group_id,batch in batches.items():
        keys=expected_batches[group_id];raw=batch['value'];sig=stable(keys)
        assert raw['keys']==keys and raw['protocol']==protocol and raw['signature']==sig,'Raw batch identity drift'
        result=raw['output']
        assert len(result['raw'])==len(result['texts'])==len(keys),'Raw batch length mismatch'
        for j,k in enumerate(keys):
            if k not in cache:continue
            v=cache[k]
            assert v['protocol']==protocol and v['fixed_batch_signature']==sig,'Cached batch contract drift'
            assert v['batch_result_sha256']==batch['sha256'],'Raw batch hash mismatch'
            assert v['raw_output']==result['raw'][j] and v['parsed_text']==result['texts'][j],'Cache/raw row mismatch'
            assert v['prediction']==predict(result['texts'][j],labels),'Cache prediction/parsed-text mismatch'
    for i,k in enumerate(order):
        if k in cache:assert str(i//2) in batches,'Cached prediction lacks raw batch'
    return dict(cached_contexts=len(cache),expected_contexts=len(order),raw_batches=len(batches),complete=require_complete)

def audit_cache(module,ds,shot,require_complete):
    out=module.ROOT/f'{shot}shot'/ds
    module.cv.verify_tree(out/'plan.freeze.json')
    plan=module.core.read(out/'test_plan.json')
    tr={x.sample_id:x for x in module.core.manifest(ds,'train')}
    query={x.sample_id:x for x in module.core.manifest(ds,'test')}
    assert [x['query_id'] for x in plan]==list(query),'Plan/query coverage drift'
    labels=sorted({x.label for x in tr.values()})
    methods=module.METHODS[:5 if shot==1 else 7]
    ph=module.core.stable([module.CONTRACT,module.core.sha(out/'plan.freeze.json')])
    records={};mapping={m:{} for m in methods}
    for row in plan:
        q=query[row['query_id']];assert set(row['methods'])==set(methods)
        for m in methods:
            demos=[tr[j] for j in row['methods'][m]]
            conversation=module.conv(ds,q,demos,labels,m,row)
            key=module.core.stable([ph,q.sample_id,conversation,q.image_sha256,[(x.sample_id,x.image_sha256) for x in demos]])
            records.setdefault(key,conversation);mapping[m][q.sample_id]=key
    cp=out/'context_cache.jsonl';cache={}
    if cp.exists():
        for line in cp.read_text(encoding='utf-8').splitlines():
            v=json.loads(line);assert v['key'] not in cache,'Duplicate cache key';cache[v['key']]=v
    batches={}
    for path in sorted((out/'batches').glob('*.json')):
        assert path.stem.isdigit(),'Unexpected batch name'
        key=str(int(path.stem));assert key not in batches
        batches[key]=dict(value=module.core.read(path),sha256=module.core.sha(path))
    result=validate_cache(list(records),cache,batches,ph,module.prediction,labels,module.core.stable,module.core.sha,require_complete)
    if require_complete:
        predictions=module.core.read(out/'predictions.json')
        assert set(predictions)==set(methods)
        for m,mp in mapping.items():
            assert set(predictions[m])==set(query)
            for qid,key in mp.items():assert predictions[m][qid]==cache[key]['prediction'],'Prediction/context mapping drift'
        saved=module.core.read(out/'results.json');assert saved['unique_contexts']==len(records)
    return result

def selfcheck(module):
    import copy
    keys=['a','b','c'];ph='synthetic-cpu-only';labels=['cat','dog'];batches={};cache={}
    for i in range(0,3,2):
        kk=keys[i:i+2];sig=module.core.stable(kk)
        b=dict(keys=kk,protocol=ph,signature=sig,output=dict(raw=['cat<eos>']*len(kk),texts=['cat']*len(kk)))
        digest=module.core.stable(b);batches[str(i//2)]=dict(value=b,sha256=digest)
        for k in kk:cache[k]=dict(key=k,protocol=ph,fixed_batch_signature=sig,batch_result_sha256=digest,raw_output='cat<eos>',parsed_text='cat',prediction='cat')
    def check(c,b,complete=True):return validate_cache(keys,c,b,ph,module.prediction,labels,module.core.stable,None,complete)
    check(cache,batches)
    partial=copy.deepcopy(cache);del partial['b'];check(partial,batches,False)
    failures=[]
    for name in ('prediction','raw_output','parsed_text','batch_result_sha256','fixed_batch_signature','protocol'):
        broken=copy.deepcopy(cache);broken['a'][name]='wrong'
        try:check(broken,batches)
        except AssertionError:failures.append(name)
        else:raise AssertionError('Corruption accepted: '+name)
    for name in ('keys','signature','protocol'):
        broken=copy.deepcopy(batches);broken['0']['value'][name]=['b','a'] if name=='keys' else 'wrong'
        try:check(cache,broken)
        except AssertionError:failures.append('raw_'+name)
        else:raise AssertionError('Raw corruption accepted: '+name)
    try:check(cache,{'0':batches['0']})
    except AssertionError:failures.append('missing_raw_batch')
    else:raise AssertionError('Missing raw accepted')
    try:check(partial,batches)
    except AssertionError:failures.append('incomplete_cache')
    else:raise AssertionError('Incomplete accepted')
    return dict(valid_full_and_partial_batch_passed=True,rejected_cases=failures,synthetic_only=True,test_access=False)
