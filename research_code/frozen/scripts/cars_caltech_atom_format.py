"""Narrow, logged compatibility for explicit abstain labels; original raw is retained."""
import json
import re
from run_qwen25_nf4_list_compat import compatible_parse as list_parse
from dgdr.hashing import stable_hash

NORMALIZATIONS = []


def normalize(raw):
    clean = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
    clean = re.sub(r'\s*```$', '', clean)
    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        return raw, []
    groups = value if isinstance(value, list) else value.get('groups', []) if isinstance(value, dict) else []
    changes = []
    if isinstance(groups, list):
        for gi, group in enumerate(groups):
            if not isinstance(group, dict) or not isinstance(group.get('states'), list):
                continue
            for si, state in enumerate(group['states']):
                if (isinstance(state, dict) and state.get('label') == 'abstain'
                        and 'coordinate' in state and state['coordinate'] is None
                        and 'abstain' not in state):
                    state['abstain'] = True
                    changes.append({'group_index': gi, 'state_index': si, 'added': {'abstain': True}})
    return (json.dumps(value, ensure_ascii=False), changes) if changes else (raw, [])


def compatible_parse(raw, *, request, source_model):
    normalized, changes = normalize(raw)
    groups = list_parse(normalized, request=request, source_model=source_model)
    if changes:
        NORMALIZATIONS.append(dict(prompt_id=request.prompt_id, raw_hash=stable_hash(raw),
            normalized_hash=stable_hash(normalized), changes=changes))
    return groups
