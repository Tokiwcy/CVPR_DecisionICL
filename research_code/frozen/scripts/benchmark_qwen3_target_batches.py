"""Validation-only GPU batching benchmark; never reads test manifests or outcomes."""
from __future__ import annotations
import argparse,gc,inspect,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_qwen3_five_dataset_recalibrated as r
from PIL import Image

class BatchedBackend(r.official.OfficialBackend):
    def forward_many(self, records, question, labels):
        opened=[]; conversations=[]
        try:
            for record in records:
                demos=[]
                for path,label in zip(record['demo_images'],record['demo_labels']):
                    with Image.open(path) as source: im=source.convert('RGB')
                    opened.append(im); demos.append((im,label))
                with Image.open(record['query_image']) as source: query=source.convert('RGB')
                opened.append(query)
                conversations.append(r.official.official_content(question,query,demos,labels,record['method'],record['attribute'],record['modified_caption']))
            self.processor.tokenizer.padding_side='left'
            inputs=self.processor.apply_chat_template(conversations,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',processor_kwargs={'padding':True}).to(self.model.device,dtype=self.torch.bfloat16)
            width=int(inputs['input_ids'].shape[1])
            start=time.perf_counter()
            with self.torch.inference_mode():
                outputs=self.model.generate(**inputs,max_new_tokens=512,do_sample=False,use_cache=True)
            self.torch.cuda.synchronize()
            raw=self.processor.batch_decode(outputs[:,width:],skip_special_tokens=True,clean_up_tokenization_spaces=False)
            lookup={label.lower():label for label in labels}
            elapsed=time.perf_counter()-start
            return [dict(prediction=lookup.get(text.strip().lower(),'__invalid_exact_output__'),raw_output=text,
                         stats=dict(seconds=elapsed/len(records),batch_seconds=elapsed,execution_batch_size=len(records),batch_row=i,
                                    input_tokens=int(inputs['attention_mask'][i].sum()),padded_tokens=width,
                                    exact_label_output=text.strip().lower() in lookup)) for i,text in enumerate(raw)]
        finally:
            for im in opened: im.close()

def records(dataset,shots,family,count):
    train=r.legacy.samples(dataset,'train'); val=r.legacy.samples(dataset,'validation')
    labels=tuple(sorted({x.label for x in train}))
    result=[]
    for i in range(count):
        q=val[(i*37)%len(val)]
        demos=[train[(i*71+j*19)%len(train)] for j in range(shots)]
        result.append(dict(query_id=q.sample_id,query_image=q.image_path,demo_ids=[x.sample_id for x in demos],demo_images=[x.image_path for x in demos],demo_labels=[x.label for x in demos],method=r.METHODS[0] if family=='baseline' else r.METHODS[5],attribute='appearance',modified_caption='An image with a different appearance.'))
    return labels,result

def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=8); args=p.parse_args()
    out=r.ROOT/'gpu_benchmark_20260910';out.mkdir(parents=True,exist_ok=True)
    backend=BatchedBackend(); reports=[]
    try:
        for dataset in r.ORDER:
            for shots,family in ((1,'baseline'),(2,'baseline'),(2,'circles')):
                labels,items=records(dataset,shots,family,args.count)
                question=r.common.source(dataset).instruction
                backend.forward(items[0],question,labels)
                values={}
                for batch in (1,2,4):
                    backend.torch.cuda.empty_cache(); backend.torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter(); outputs=[]
                    try:
                        for i in range(0,len(items),batch):
                            if batch==1: outputs.append(backend.forward(items[i],question,labels))
                            else: outputs.extend(backend.forward_many(items[i:i+batch],question,labels))
                        values[str(batch)]=dict(status='passed',seconds=time.perf_counter()-start,peak_mib=backend.torch.cuda.max_memory_allocated()/2**20,outputs=outputs)
                    except backend.torch.OutOfMemoryError as error:
                        values[str(batch)]=dict(status='oom',message=str(error)[:160]); gc.collect();backend.torch.cuda.empty_cache()
                        break
                baseline=values['1']
                for batch,v in values.items():
                    if v['status']=='passed':
                        v['raw_equal']=all(a['raw_output']==b['raw_output'] for a,b in zip(baseline['outputs'],v['outputs']))
                        v['prediction_equal']=all(a['prediction']==b['prediction'] for a,b in zip(baseline['outputs'],v['outputs']))
                        v['speedup']=baseline['seconds']/v['seconds']
                report=dict(dataset=dataset,shots=shots,family=family,count=len(items),values=values,validation_ids=[x['query_id'] for x in items])
                reports.append(report)
                r.write(out/'benchmark.json',dict(validation_only=True,test_manifest_loaded=False,reports=reports,source_sha256=r.file_sha256(__file__)))
                print(json.dumps({k:v for k,v in report.items() if k not in ('values','validation_ids')}|{'batches':{k:{kk:vv for kk,vv in v.items() if kk!='outputs'} for k,v in values.items()}}),flush=True)
    finally:
        del backend;gc.collect()
        import torch;torch.cuda.empty_cache()
if __name__=='__main__': main()
