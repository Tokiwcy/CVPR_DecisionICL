"""Qwen2.5-VL-7B NF4 double quant, BF16 compute, clean-room target contract."""
from pathlib import Path
import copy
import gc
import time
import torch
from PIL import Image
from run_qwen3_dataset_plan import single_token_codes, conversation

MODEL = Path(__file__).resolve().parents[2] / 'models/Qwen2.5-VL-7B-Instruct'
MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
REVISION = 'local-checkpoint-all-files-sha256-frozen'
CONTRACT = dict(model=MODEL_ID,revision=REVISION,model_path=str(MODEL),
    dtype='bfloat16',quantization='nf4',double_quant=True,compute_dtype='bfloat16',
    attention='sdpa',device='cuda:0',processor_settings=dict(min_pixels=65536,max_pixels=401408),
    conversation_version='qwen25-cleanroom-1shot-v1',
    classification='constrained next-token single-token class-code logits',
    class_code_policy='common cleanroom single CJK token codes; not legacy alphabetic codes',
    shots=1,batch_size=1,cpu_threads=4,use_cache_classification=False,
    discovery_max_new_tokens=3072,discovery_do_sample=False,
    execution='complete one dataset before next; isolated GPU stage processes')

class Qwen25Backend:
    model_id=MODEL_ID+':local-sha256-frozen:nf4-double-bf16'
    def __init__(self):
        from transformers import AutoProcessor,BitsAndBytesConfig,Qwen2_5_VLForConditionalGeneration
        from bitsandbytes.nn import Linear4bit
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        self.processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,**CONTRACT['processor_settings'])
        self.processor.tokenizer.padding_side='left'; self.tokenizer=self.processor.tokenizer
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,
            quantization_config=quant,device_map={'':'cuda:0'},dtype=torch.bfloat16,attn_implementation='sdpa').eval()
        if any(p.device.type!='cuda' for p in self.model.parameters()): raise RuntimeError('Offload forbidden')
        quantized=[]
        for name,module in self.model.named_modules():
            if isinstance(module,Linear4bit):
                w=module.weight; state=w.quant_state
                if w.dtype!=torch.uint8 or state is None or state.quant_type!='nf4' or not state.nested:
                    raise RuntimeError('Incorrect quantized weight: '+name)
                quantized.append(name)
        if not quantized: raise RuntimeError('No NF4 modules found')
        self.audit=dict(model_class=type(self.model).__name__,quantized_module_count=len(quantized),
            quantized_modules=quantized,cpu_offload=False,allocated_gib=torch.cuda.memory_allocated()/2**30,
            image_processor=self.processor.image_processor.to_dict())

    def class_codes(self,count): return single_token_codes(self.tokenizer,count)
    def conversation(self,instruction,query_image,demo_image,demo_label,labels,codes):
        if demo_label not in labels: raise ValueError('Unknown demo label')
        return conversation(instruction,query_image,demo_image,demo_label,labels,codes)
    def inputs(self,messages):
        opened=[]; prepared=copy.deepcopy(messages)
        try:
            for message in prepared:
                for item in message['content']:
                    if item['type']=='image':
                        with Image.open(item['image']) as source: im=source.convert('RGB')
                        opened.append(im); item['image']=im
            return self.processor.apply_chat_template([prepared],add_generation_prompt=True,tokenize=True,
                return_dict=True,return_tensors='pt',processor_kwargs={'padding':True}).to(self.model.device,dtype=torch.bfloat16)
        finally:
            for im in opened: im.close()
    def classify(self,messages,token_ids):
        if sum(i['type']=='image' for m in messages for i in m['content'])!=2:
            raise ValueError('Exactly one demo and one query required')
        start=time.perf_counter(); inputs=self.inputs(messages)
        with torch.inference_mode():
            logits=self.model(**inputs,logits_to_keep=1,use_cache=False).logits[0,-1,token_ids].float()
        if not torch.isfinite(logits).all(): raise RuntimeError('Nonfinite classification')
        values=logits.cpu().tolist()
        return int(logits.argmax()),values,dict(seconds=time.perf_counter()-start,
            tokens=inputs['input_ids'].shape[1],pixel_shape=list(inputs['pixel_values'].shape),peak_gib=torch.cuda.max_memory_allocated()/2**30)
    def generate(self,image_paths,prompt):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        inputs=self.inputs([dict(role='user',content=[*[dict(type='image',image=str(p)) for p in image_paths],dict(type='text',text=prompt)])])
        start=time.perf_counter()
        with torch.inference_mode():
            output=self.model.generate(**inputs,do_sample=False,max_new_tokens=CONTRACT['discovery_max_new_tokens'])
        tokens=output[0,inputs['input_ids'].shape[1]:].tolist()
        self.last_generation_stats=dict(seconds=time.perf_counter()-start,new_tokens=len(tokens),
            input_tokens=inputs['input_ids'].shape[1],peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            reserved_gib=torch.cuda.memory_reserved()/2**30)
        return self.tokenizer.decode(tokens,skip_special_tokens=True).strip()
