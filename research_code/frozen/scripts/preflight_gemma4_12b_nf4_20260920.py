"""Pinned download and train/validation-only Gemma 4 12B NF4 preflight.

This entry point never starts a test run or reads a test manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

SCRIPT = Path(__file__).resolve()
CLEAN = SCRIPT.parents[1]
REPO = CLEAN.parent
OUT = CLEAN / 'artifacts/priority_1shot_ablation_20260920/gemma_preflight'
MODEL = REPO / 'models/gemma-4-12B-it'
MODEL_ID = 'google/gemma-4-12B-it'
REVISION = '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'
DOC = CLEAN / 'docs/priority_1shot_ablation_20260920_zh.md'
ORDER = ('dtd', 'aircraft', 'cub', 'dogs', 'flowers102')
CONTRACT = dict(model=MODEL_ID, revision=REVISION, quantization='bitsandbytes-nf4',
                double_quant=True, compute_dtype='bfloat16', device='cuda:0',
                cpu_offload=False, attention='sdpa', enable_thinking=False,
                max_soft_tokens=280, target_batch=2, shots=1,
                classification_max_new_tokens=512, discovery_max_new_tokens=3072,
                output_parsing='official processor.parse_response content; strip/lower exact match')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path[:0] = [str(SCRIPT.parent), str(CLEAN / 'src')]


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def emit(stage, **kwargs):
    value = dict(stage=stage, time=time.time(), pid=os.getpid(), **kwargs)
    write(OUT / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def held():
    for root in (OUT, OUT.parent):
        for name in ('PAUSED_BY_USER.json', 'DIAGNOSTIC_HOLD.json', 'WATCHDOG_STOP'):
            if (root / name).exists():
                raise RuntimeError(f'preflight held: {root / name}')


def manifest(dataset, split):
    if split not in ('train', 'validation'):
        raise ValueError('preflight forbids test access')
    if dataset == 'flowers102':
        return CLEAN / 'artifacts/qwen3_five_dataset_recalibrated_20260910/gpu_batch2_flowers102_20260911/data' / (split + '.jsonl')
    if dataset == 'cub':
        return CLEAN / 'artifacts/cub_full_current_atoms_20260904' / f'cub.{split}.jsonl'
    return CLEAN / f'artifacts/four_dataset_atom_label_20260905/{dataset}' / f'{dataset}.{split}.jsonl'


def dataset_inputs(dataset):
    from dgdr.io import load_manifest
    train, val = (load_manifest(manifest(dataset, split), expected_split=split) for split in ('train', 'validation'))
    assert not {x.image_sha256 for x in train}.intersection(x.image_sha256 for x in val)
    if dataset == 'cub':
        question = 'Identify the bird species shown in the image.'
    else:
        question = read(manifest(dataset, 'train').parent / 'dataset_config.json')['instruction']
    return train, val, question


def cpu_check():
    from transformers import AutoConfig, AutoProcessor
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    assert config.model_type == 'gemma4_unified'
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    assert processor.image_processor.max_soft_tokens == 280
    reports = []
    for dataset in ORDER:
        train, val, question = dataset_inputs(dataset)
        reports.append(dict(dataset=dataset, train=len(train), validation=len(val),
                            question=question, files={str(manifest(dataset, s)): sha(manifest(dataset, s)) for s in ('train', 'validation')}))
    files = {str(p): sha(p) for p in (SCRIPT, DOC, MODEL / 'config.json', MODEL / 'processor_config.json', MODEL / 'tokenizer_config.json', MODEL / 'chat_template.jinja')}
    write(OUT / 'cpu_preflight.json', dict(status='passed', contract=CONTRACT, reports=reports,
          test_manifest_loaded=False, files=files))
    emit('cpu_preflight_passed')


def download():
    from huggingface_hub import HfApi, snapshot_download
    held()
    receipt = OUT / 'download.complete.json'
    if receipt.exists():
        saved = read(receipt)
        assert saved['revision'] == REVISION
        for path, digest in saved['files'].items():
            assert sha(path) == digest, f'model file drift: {path}'
        return
    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    weights = next(x for x in info.siblings if x.rfilename == 'model.safetensors')
    expected = weights.lfs.sha256
    write(OUT / 'download.expected.json', dict(model=MODEL_ID, revision=REVISION,
          weight_bytes=weights.size, weight_sha256=expected, contract=CONTRACT))
    emit('downloading', bytes=weights.size, revision=REVISION)
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=MODEL,
                      allow_patterns=['*.json', '*.jinja', '*.model', '*.safetensors'], max_workers=4)
    held()
    emit('verifying_download')
    files = {str(p): sha(p) for p in MODEL.iterdir() if p.is_file() and p.suffix in ('.json', '.jinja', '.model', '.safetensors')}
    assert files[str(MODEL / 'model.safetensors')] == expected, 'official weight SHA mismatch'
    write(receipt, dict(status='passed', model=MODEL_ID, revision=REVISION, files=files,
                       official_weight_sha256=expected))
    emit('download_verified')


def final_text(parsed):
    if not isinstance(parsed, dict) or parsed.get('role') != 'assistant':
        raise ValueError(f'unexpected official response schema: {parsed!r}')
    content = parsed.get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(x, dict) and x.get('type') == 'text' for x in content):
        return ''.join(x['text'] for x in content)
    raise ValueError(f'unexpected official content schema: {parsed!r}')


def gpu_check():
    import msvcrt
    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    from PIL import Image
    from dgdr.discovery import make_discovery_requests
    from run_qwen3_circles_official_prompt_2shot_four import official_content, METHODS

    held()
    saved = read(OUT / 'cpu_preflight.json')
    for path, digest in saved['files'].items():
        assert sha(path) == digest, f'preflight source drift: {path}'
    lock_path = CLEAN / 'artifacts/qwen3_five_dataset_recalibrated_20260910/gpu_queue.lock'
    lock = lock_path.open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.94, 0)
    emit('loading_gemma12b_nf4')
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL, local_files_only=True,
        quantization_config=quantization, device_map={'': 'cuda:0'}, dtype=torch.bfloat16,
        attn_implementation='sdpa').eval()
    assert getattr(model, 'is_loaded_in_4bit', False), '4bit not active'
    assert {p.device.type for p in model.parameters()} == {'cuda'}, 'offload forbidden'

    def forward(conversations, limit):
        held()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        inputs = processor.apply_chat_template(conversations, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors='pt', padding=True,
            enable_thinking=False).to('cuda:0', dtype=torch.bfloat16)
        width = inputs['input_ids'].shape[-1]
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=limit, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        parsed = processor.parse_response(output[:, width:], prefix=inputs['input_ids'])
        if isinstance(parsed, dict):
            parsed = [parsed]
        texts = [final_text(x) for x in parsed]
        assert len(texts) == len(conversations) and all(x.strip() for x in texts)
        value = dict(texts=texts, parsed=parsed,
            raw=processor.batch_decode(output[:, width:], skip_special_tokens=False),
            input_tokens=inputs['attention_mask'].sum(-1).tolist(),
            seconds=time.perf_counter()-started,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        del inputs, output
        return value

    reports = []
    for dataset in ORDER:
        train, val, question = dataset_inputs(dataset)
        labels = sorted({x.label for x in train})
        def area(sample):
            with Image.open(sample.image_path) as im:
                return im.width * im.height
        queries = sorted(val, key=lambda x: (-area(x), x.sample_id))[:4]
        demos = sorted(train, key=lambda x: (-area(x), x.sample_id))[:2]
        for x in queries + demos:
            assert sha(x.image_path) == x.image_sha256, 'preflight image bytes changed'
        conversations = [official_content(question, x.image_path,
            [(demos[i % 2].image_path, demos[i % 2].label)], labels, METHODS[0]) for i, x in enumerate(queries[:3])]
        emit('validation_classification', dataset=dataset)
        pair = forward(conversations[:2], 512)
        repeat = forward(conversations[:2], 512)
        assert pair['raw'] == repeat['raw'], 'fixed batch repeat drift'
        tail = forward(conversations[2:], 512)
        requests = make_discovery_requests(sample_ids=[x.sample_id for x in queries],
            image_paths=[x.image_path for x in queries], task_instruction=question,
            class_names=labels, batch_size=4, max_groups_per_perspective=6)
        request = requests[0]
        discovery = [[dict(role='user', content=[dict(type='image', image=p) for p in request.image_paths]
                          + [dict(type='text', text=request.prompt)])]]
        emit('validation_discovery_load', dataset=dataset)
        discovered = forward(discovery, 3072)
        # Discovery content is discarded scientifically; this checks hardware/format only.
        from cars_caltech_atom_format import compatible_parse
        groups = compatible_parse(discovered['texts'][0], request=request, source_model=MODEL_ID)
        assert groups, 'discovery schema returned no groups'
        row = dict(dataset=dataset, query_ids=[x.sample_id for x in queries],
            train_ids=[x.sample_id for x in demos], pair=pair, tail=tail,
            repeat_raw_equal=True, discovery=discovered, discovery_groups=len(groups),
            strict_valid=[x.strip().lower() in {y.lower() for y in labels} for x in pair['texts']+tail['texts']],
            test_manifest_loaded=False, used_for_factor_selection=False)
        reports.append(row)
        write(OUT / f'{dataset}.gpu_preflight.json', row)
    files = {str(p): sha(p) for p in [SCRIPT, DOC, OUT/'cpu_preflight.json', OUT/'download.complete.json']}
    write(OUT/'complete.json', dict(status='passed', contract=CONTRACT, reports=reports,
        files=files, transformers=transformers.__version__, torch=torch.__version__,
        quantized=True, test_manifest_loaded=False, no_test_inference=True))
    emit('gemma_preflight_complete', next_stage='qwen_source_ablation')
    del model
    torch.cuda.empty_cache()
    lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('cpu-check', 'download', 'gpu-check', 'all'))
    args = parser.parse_args()
    if args.stage in ('download', 'all'):
        download()
    if args.stage in ('cpu-check', 'all'):
        cpu_check()
    if args.stage in ('gpu-check', 'all'):
        gpu_check()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        emit('blocked', error_type=type(exc).__name__, error=str(exc))
        write(OUT/'DIAGNOSTIC_HOLD.json', dict(error_type=type(exc).__name__, error=str(exc), time=time.time()))
        raise
