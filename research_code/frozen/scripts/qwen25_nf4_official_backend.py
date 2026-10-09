"""Qwen2.5-VL-7B NF4 double quant, BF16 compute, official free generation."""
from pathlib import Path
import torch
from qwen25_nf4_backend import Qwen25Backend, MODEL, MODEL_ID, CONTRACT as BASE_CONTRACT
PLACEMENT_CONTRACT={'':'cuda:0'}
CONTRACT={**BASE_CONTRACT,'quantization':'nf4','double_quant':True,'compute_dtype':'bfloat16','device':'cuda:0','device_map_policy':PLACEMENT_CONTRACT,'conversation_version':'qwen25-nf4-official-classification-v1','classification':'official free-generation exact category match','class_code_policy':'not used','shots':[1,2],'batch_size':2,'use_cache_classification':True,'execution':'serial five datasets, fixed batch2 target inference'}

def initialize(self):
    torch.cuda.set_per_process_memory_fraction(.94,0)
    Qwen25Backend.__init__(self)
    self.torch=torch
    # All parameters were verified CUDA by the underlying NF4 constructor.
    # Some Transformers full-device loads omit the Accelerate metadata attribute.
    if not hasattr(self.model,'hf_device_map'):self.model.hf_device_map={'':'cuda:0'}
    if not getattr(self.model,'is_quantized',False):raise RuntimeError('Expected NF4 quantization')
    self.audit.update(quantization='nf4',double_quant=True,compute_dtype='bfloat16',device_map=self.model.hf_device_map,placement_policy=PLACEMENT_CONTRACT)

class NF4Backend(Qwen25Backend):
    model_id=MODEL_ID+':nf4-double-bf16-official'
    __init__=initialize
