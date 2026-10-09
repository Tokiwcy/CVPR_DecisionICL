"""Target batch2 unchanged; CIRCLES descriptions use a train-qualified batch."""
from pathlib import Path
import run_gemma12_main_eval_v4_20260926 as audited
import run_gemma12_circles_validation_batched_v4_20260926 as accelerated
import audit_gemma12_main_eval_v5_20260926 as replay


def main():
    size=accelerated.configure()
    base=audited.base
    base.ROOT=base.prep.ROOT.parent/'evaluation_v5_batched_descriptions'
    base.CONTRACT=dict(base.CONTRACT,description_batch=size,
        circles_validation_root=str(base.cv.ROOT),
        description_contract=base.cv.CONTRACT,
        speed_selection_sha256=base.core.sha(accelerated.SPEED/'selection.json'))
    base.audit_plan=replay.audit_plan
    audited.main()


if __name__=='__main__':main()
