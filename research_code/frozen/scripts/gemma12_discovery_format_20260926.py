"""Narrow Gemma12 JSON key compatibility, validated without test access.
Only restore a unique corrupt applies_ key; never invent a missing value.
"""
import json,re
from cars_caltech_atom_format import compatible_parse as prior_parse
from dgdr.hashing import stable_hash
NORMALIZATIONS=[]
EXPECTED={'region','attribute','question','applies_to','states'}
def normalize(raw):
    clean=re.sub(r'^```(?:json)?\s*','',raw.strip(),flags=re.I)
    clean=re.sub(r'\s*```$','',clean)
    try:value=json.loads(clean)
    except json.JSONDecodeError:return raw,[]
    groups=value.get('groups',[]) if isinstance(value,dict) else value if isinstance(value,list) else []
    changes=[]
    for i,g in enumerate(groups):
        if not isinstance(g,dict) or 'applies_to' in g:continue
        missing=EXPECTED-set(g);extras=set(g)-EXPECTED
        if missing=={'applies_to'} and len(extras)==1:
            key=next(iter(extras))
            if key.startswith('applies_') and isinstance(g[key],str) and g[key].strip():
                original=g.pop(key);g['applies_to']=original
                changes.append(dict(group_index=i,old_key=key,new_key='applies_to',value=original))
    return (json.dumps(value,ensure_ascii=False),changes) if changes else (raw,[])
def compatible_parse(raw,*,request,source_model):
    normalized,changes=normalize(raw)
    groups=prior_parse(normalized,request=request,source_model=source_model)
    if changes:NORMALIZATIONS.append(dict(prompt_id=request.prompt_id,raw_hash=stable_hash(raw),normalized_hash=stable_hash(normalized),changes=changes))
    return groups
