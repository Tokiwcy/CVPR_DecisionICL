"""Selected evidence policy: native SigLIP logits, no train mean subtraction.

For future target-discovered banks, call build_raw_siglip_evidence separately on
each split. Retrieval weights are deliberately outside this module and must be
relearned for the exact new bank/method. Historical calibrated files are immutable.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dgdr.evidence import build_evidence_store, UNCALIBRATED_EVIDENCE_TRANSFORM

POLICY={
    "evidence_encoder":"SigLIP2 Base patch16-224",
    "state_logit_bias":"none",
    "text_template":"This image shows {description}.",
    "state_probability":"group softmax of native raw logits",
    "signed_margin":"unchanged state versus log-mean-exp of other states",
    "template_ensemble":False,
    "confidence_gate":False,
    "description_refinement":False,
    "factor_support_change":False,
    "retrieval_weights":"independently relearn for each exact method and target-discovered bank",
    "semantic_limit":"relative compatibility, not guaranteed absolute attribute presence",
}

def build_raw_siglip_evidence(*,samples,image_embeddings,bank,encoder):
    if any(s.split not in ("train","validation","test") for s in samples):
        raise ValueError("unexpected evidence split")
    if len({s.split for s in samples})!=1:
        raise ValueError("do not combine evidence splits")
    if "siglip" not in encoder.model_id.lower():
        raise ValueError("this selected policy only supports SigLIP")
    result=build_evidence_store(samples=samples,image_embeddings=image_embeddings,bank=bank,
        encoder=encoder,text_template=POLICY["text_template"],calibration=None)
    validate_raw_policy(result)
    return result

def validate_raw_policy(store):
    store.validate()
    if store.calibration_hash!="identity" or store.transform!=UNCALIBRATED_EVIDENCE_TRANSFORM:
        raise ValueError("selected SigLIP policy forbids train mean debias or a different transform")
    if store.text_template!=POLICY["text_template"]:
        raise ValueError("selected SigLIP policy forbids a different prompt template")
