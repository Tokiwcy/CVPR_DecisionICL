"""Gemma4 12B NF4 backend. Caller must hold the shared GPU queue lock.
The pread weight reader avoids the diagnosed Windows mmap storage crash.
"""
from pathlib import Path
import time
import preflight_gemma4_12b_nf4_20260920 as spec
MODEL=spec.MODEL
MODEL_ID=spec.MODEL_ID
CONTRACT=dict(spec.CONTRACT,shots=[1,2],weight_reader='safetensors pread',
    quantization='bitsandbytes-nf4',cpu_offload=False,output_parsing='trim per-row at first generation EOS inclusive; official parse_response; exact label; no semantic cleanup')

def install_pread():
    import transformers.modeling_utils as modeling
    original=modeling.safe_open
    if getattr(original,'_gemma12_pread',False): return
    def reader(filename,*args,**kwargs):
        if Path(filename).resolve()==(MODEL/'model.safetensors').resolve():
            kwargs['backend']='pread'
        return original(filename,*args,**kwargs)
    reader._gemma12_pread=True
    modeling.safe_open=reader

def parse_generated(processor, generated_ids, prefix_ids, eos_token_ids):
    eos={int(x) for x in eos_token_ids}
    parsed=[]; lengths=[]
    for ids,prefix in zip(generated_ids,prefix_ids):
        values=ids.tolist() if hasattr(ids,'tolist') else list(ids)
        end=next((i+1 for i,x in enumerate(values) if x in eos),len(values))
        # Drop ONLY transport padding after generation termination, never words.
        parsed.append(processor.parse_response(values[:end],prefix=prefix))
        lengths.append(end)
    return parsed,lengths

class Gemma12Backend:
    model_id=MODEL_ID
    def __init__(self):
        import torch
        from transformers import AutoProcessor,AutoModelForMultimodalLM,BitsAndBytesConfig
        install_pread()
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(.94,0)
        self.torch=torch
        self.processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        assert self.processor.image_processor.max_soft_tokens==280
        self.processor.tokenizer.padding_side='left'
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        self.model=AutoModelForMultimodalLM.from_pretrained(MODEL,local_files_only=True,
            quantization_config=quant,device_map={'':'cuda:0'},dtype=torch.bfloat16,
            attn_implementation='sdpa').eval()
        assert self.model.is_loaded_in_4bit
        assert {p.device.type for p in self.model.parameters()}=={'cuda'}
        self.audit=dict(contract=CONTRACT,model_class=type(self.model).__name__,
            processor_class=type(self.processor).__name__,quantized=True,cpu_offload=False)
    def forward(self,conversations,limit=512):
        torch=self.torch
        torch.cuda.reset_peak_memory_stats()
        inputs=self.processor.apply_chat_template(conversations,add_generation_prompt=True,
            tokenize=True,return_dict=True,return_tensors='pt',processor_kwargs={'padding':True},
            enable_thinking=False).to('cuda:0',dtype=torch.bfloat16)
        width=inputs['input_ids'].shape[-1]
        start=time.perf_counter()
        with torch.inference_mode():
            output=self.model.generate(**inputs,max_new_tokens=limit,do_sample=False,use_cache=True)
        torch.cuda.synchronize()
        parsed,lengths=parse_generated(self.processor,output[:,width:],inputs['input_ids'],self.model.generation_config.eos_token_id)
        texts=[spec.final_text(x) for x in parsed]
        assert len(texts)==len(conversations)
        self.last_generation_stats=dict(seconds=time.perf_counter()-start,
            input_tokens=inputs['attention_mask'].sum(-1).tolist(),new_tokens=int(output.shape[1]-width),
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,execution_batch_size=len(conversations))
        value=dict(texts=texts,parsed=parsed,raw=self.processor.batch_decode(output[:,width:],skip_special_tokens=False),
            stats=self.last_generation_stats,generated_lengths=lengths)
        del inputs,output
        return value
    def generate(self,image_paths,prompt):
        conv=[dict(role='user',content=[dict(type='image',image=str(p)) for p in image_paths]
            +[dict(type='text',text=prompt)])]
        return self.forward([conv],3072)['texts'][0]
