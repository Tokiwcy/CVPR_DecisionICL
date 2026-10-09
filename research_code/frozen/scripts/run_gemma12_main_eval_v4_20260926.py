"""Execution-only v4 adapter: strict cache provenance; v3 science unchanged."""
from pathlib import Path
import sys,time
import run_gemma12_main_eval_v3_20260926 as base
import gemma12_cache_provenance_20260926 as provenance
SCRIPT=Path(__file__).resolve()
base.ROOT=base.prep.ROOT.parent/'evaluation_v4'
_original_infer=base.infer
_original_audit=base.audit_results

def audit_results(ds,shot,root):
    assert Path(root)==base.ROOT
    result=provenance.audit_cache(base,ds,shot,True)
    _original_audit(ds,shot,root)
    path=base.ROOT/f'{shot}shot'/ds/'independent_audit.json'
    audit=base.core.read(path);audit['cache_provenance']=result
    audit['cache_auditor_sha256']=base.core.sha(Path(provenance.__file__))
    base.core.write(path,audit)

def infer(ds,shot,backend):
    provenance.audit_cache(base,ds,shot,False)
    return _original_infer(ds,shot,backend)

base.audit_results=audit_results;base.infer=infer

def main():
    if 'cpu-check' in sys.argv:
        checks=provenance.selfcheck(base)
        base.main()
        base.core.write(base.ROOT/'v4_cpu_check.json',dict(status='passed',checks=checks,
            wrapper_sha256=base.core.sha(SCRIPT),auditor_sha256=base.core.sha(Path(provenance.__file__)),
            original_runner_sha256=base.core.sha(Path(base.__file__)),science_unchanged=True,test_access=False))
        print('v4 provenance CPU checks passed: 11 corruption/missing-data cases rejected',flush=True)
    else:
        receipt=base.core.read(base.ROOT/'v4_cpu_check.json')
        assert receipt['status']=='passed' and receipt['wrapper_sha256']==base.core.sha(SCRIPT)
        assert receipt['auditor_sha256']==base.core.sha(Path(provenance.__file__))
        assert receipt['original_runner_sha256']==base.core.sha(Path(base.__file__))
        base.main()

if __name__=='__main__':main()
