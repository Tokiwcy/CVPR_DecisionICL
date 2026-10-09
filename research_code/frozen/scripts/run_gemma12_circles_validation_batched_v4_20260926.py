"""Separately frozen CIRCLES description batching, selected on train only."""
from pathlib import Path
import run_gemma12_circles_validation_v3_20260926 as cv
SCRIPT=Path(__file__).resolve()
SPEED=cv.prep.ROOT.parent/'circles_speed_preflight'


def configure():
    cv.verify_tree(SPEED/'complete.json')
    selection=cv.core.read(SPEED/'selection.json')
    assert selection['status']=='passed' and selection['test_access'] is False
    size=selection['selected_batch']
    assert size in (4,8), 'No acceleration qualified; retain original batch2 checkpoint'
    cv.ROOT=cv.prep.ROOT.parent/'circles_validation_batched_v4'
    cv.SCRIPT=SCRIPT
    cv.CONTRACT=dict(cv.CONTRACT,description_batch=size,
        selection_receipt_sha256=cv.core.sha(SPEED/'selection.json'),
        description_cache_policy='new root; no old batch2 descriptions reused',
        independent_circles_refit=True,classification_batch_unchanged=2)
    cv.batches=lambda rows:[rows[i:i+size] for i in range(0,len(rows),size)]
    original_dependencies=cv.cal.dependencies
    cv.cal.dependencies=lambda:sorted(set([p for p in original_dependencies() if p.is_file()]+[SCRIPT,SCRIPT.parents[1]/'docs/gemma12_circles_batch_acceleration_20260926_zh.md',SPEED/'complete.json',SPEED/'selection.json']))
    return size


def cpu_check():
    cv.initialize()
    size=cv.CONTRACT['description_batch']
    assert [len(x) for x in cv.batches(list(range(2*size+1)))]==[size,size,1]
    assert list(sum(cv.batches(list(range(2*size+1))),[]))==list(range(2*size+1))
    for ds in cv.core.ORDER:
        rows=cv.core.manifest(ds,'validation');q=cv.core.common.source(ds).instruction
        assert cv.attribute_conversation(rows[0],q)[0]['content'][1]['type']=='image'
        assert len(cv.caption_conversation(rows[0],'shape')[0]['content'])==3
    cv.core.write(cv.ROOT/'cpu_check.json',dict(status='passed',script_sha256=cv.core.sha(SCRIPT),
        datasets=5,description_batch=size,queries=sum(len(cv.core.manifest(ds,'validation')) for ds in cv.core.ORDER),
        batch_order_and_tail_checks=True,test_access=False))
    print('Separate batched validation CPU contracts passed',flush=True)


def main():
    configure()
    cv.cpu_check=cpu_check
    cv.main()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        cv.emit('failed',error_type=type(exc).__name__,error=str(exc))
        cv.core.write(cv.ROOT/'DIAGNOSTIC_HOLD.json',dict(error=str(exc)))
        raise
