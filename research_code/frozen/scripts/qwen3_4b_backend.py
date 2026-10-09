"""Qwen3-VL-4B BF16 adapter; common clean-room single-token 1-shot interface."""
from pathlib import Path
import torch
from qwen25_nf4_backend import Qwen25Backend, CONTRACT as COMMON_CONTRACT

MODEL=Path(__file__).resolve().parents[2]/'models/Qwen3-VL-4B-Instruct'
MODEL_ID='Qwen/Qwen3-VL-4B-Instruct'
REVISION='local-checkpoint-all-files-sha256-frozen'
CONTRACT={**COMMON_CONTRACT,'model':MODEL_ID,'model_path':str(MODEL),
    'quantization':'none','double_quant':False,
    'conversation_version':'qwen3-cleanroom-1shot-v1'}
class Qwen3Backend(Qwen25Backend):
    model_id=MODEL_ID+':local-sha256-frozen:bf16'
    def __init__(self):
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        self.processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,**CONTRACT['processor_settings'])
        self.processor.tokenizer.padding_side='left'
        self.tokenizer=self.processor.tokenizer
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,
            device_map={'':'cuda:0'},dtype=torch.bfloat16,attn_implementation='sdpa').eval()
        if any(p.device.type!='cuda' for p in self.model.parameters()):raise RuntimeError('Offload forbidden')
        if any(p.is_floating_point() and p.dtype!=torch.bfloat16 for p in self.model.parameters()):
            raise RuntimeError('Unexpected model parameter dtype')
        self.audit=dict(model_class=type(self.model).__name__,quantized_module_count=0,
            cpu_offload=False,allocated_gib=torch.cuda.memory_allocated()/2**30,
            image_processor=self.processor.image_processor.to_dict())
        assert CONTRACT['discovery_max_new_tokens']==COMMON_CONTRACT['discovery_max_new_tokens']
