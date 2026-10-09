"""Maintain complete-only experiment tables; independent of frozen experiment code."""
from pathlib import Path
import argparse,hashlib,json,os,time
ROOT=Path(__file__).resolve().parents[1]/'artifacts/qwen3_five_dataset_recalibrated_20260910'
DATASETS=('dtd','caltech101','aircraft','cub','dogs')
LABELS=('DTD','Caltech101','Aircraft','CUB','Dogs')
METHODS=('random_official_prompt','rices_clip_official_prompt','muier_classification_adapted_siglip_official_prompt','rices_clip_atom_class_equal_raw_official_prompt','muier_classification_adapted_siglip_atom_class_equal_raw_official_prompt','circles_official_prompt','circles_atom_class_equal_raw_official_prompt')
SHORT=('Random','RICES','MUIER','RICES＋Atom','MUIER＋Atom','CIRCLES','CIRCLES＋Atom')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_atomic(p,text):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(text,encoding='utf-8');os.replace(temp,p)
def render():
    locations=read(ROOT/'execution_locations.json') if (ROOT/'execution_locations.json').exists() else {}
    status=read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
    current=locations.get('_active_root',str(ROOT))
    active=Path(current)
    if active!=ROOT and (active/'status.json').exists(): status=read(active/'status.json')
    hold=any((root/name).exists() for root in {ROOT,active} for name in ('DIAGNOSTIC_HOLD.json','PAUSED_BY_USER.json','WATCHDOG_STOP'))
    results={};state={};evidence={};errors=[]
    for dataset in DATASETS:
        for shot in (1,2):
            key=f'{dataset}/{shot}'
            out=Path(locations.get(key,str(ROOT/f'{shot}shot'/dataset)))
            state[key]='待运行'
            if status.get('dataset')==dataset and status.get('shots')==shot:
                state[key]='加速预检暂停' if hold else '运行中'
            try:
                if (out/'complete.json').exists() and (out/'independent_audit.json').exists():
                    audit=read(out/'independent_audit.json')
                    if audit['status']!='passed' or any(sha(p)!=h for p,h in audit['files'].items()):raise ValueError('独立核验凭据不一致')
                    complete=read(out/'complete.json')
                    if any(sha(p)!=h for p,h in complete['files'].items()):raise ValueError('结果哈希不一致')
                    result=read(out/'results.json')
                    expected=set(METHODS[:5] if shot==1 else METHODS)
                    if result['status']!='complete' or set(result['methods'])!=expected:raise ValueError('方法覆盖不完整')
                    results[key]=result;state[key]='已完成';evidence[key]=str(out/'independent_audit.json')
            except Exception as exc:
                state[key]='核验异常';errors.append(f'{key}: {exc}')
    lines=['# Qwen3-VL-4B 重跑累计结果','',f"更新时间：{time.strftime('%Y-%m-%d %H:%M:%S')}（本机时间）",'',f'已完成并核验：{len(results)}/10 个数据集×shot块，{sum(len(x["methods"]) for x in results.values())}/60 个方法结果。', '', '表内为完整 test Accuracy（%）；— 表示尚未完成。RICES 使用 CLIP，MUIER 为 SigLIP 分类适配版；Atom 均为类别等权 raw，各分支独立重校准。']
    for shot in (1,2):
        n=5 if shot==1 else 7
        lines += ['',f'## {shot}-shot','', '| 数据集 | '+' | '.join(SHORT[:n])+' | 状态 |','|---|'+'---:|'*n+'---|']
        for dataset,label in zip(DATASETS,LABELS):
            key=f'{dataset}/{shot}';res=results.get(key)
            cells=[f"{res['methods'][m]['metrics']['accuracy']*100:.3f}" if res else '—' for m in METHODS[:n]]
            lines.append('| '+label+' | '+' | '.join(cells)+' | '+state[key]+' |')
    lines += ['','## 已完成 Atom 分支的配对结果','','| 数据集 | Shot | 方法 | Δ Accuracy (pp) | Rescue | Harm |','|---|---:|---|---:|---:|---:|']
    for key,res in results.items():
        for m in METHODS:
            item=res['methods'].get(m,{})
            if 'paired_vs_parent' not in item:continue
            pair=item['paired_vs_parent'];d,shot=key.split('/')
            lines.append(f"| {LABELS[DATASETS.index(d)]} | {shot} | {SHORT[METHODS.index(m)]} | {pair['accuracy_delta']*100:+.3f} | {pair['rescue']} | {pair['harm']} |")
    if not results:lines+=['| 尚无完整结果 | — | — | — | — | — |']
    lines+=['','完整 Macro、bootstrap 95% CI、McNemar 及逐项凭据位于各数据集 results.json 与 independent_audit.json。仅报告结果，不据 test 调整参数或方法。','']
    if errors:lines+=['核验异常：']+errors
    write_atomic(ROOT/'live_results_zh.md','\n'.join(lines))
    write_atomic(ROOT/'live_results.json',json.dumps(dict(updated_at=time.time(),completed_blocks=len(results),completed_methods=sum(len(x['methods']) for x in results.values()),states=state,results=results,evidence=evidence,errors=errors),ensure_ascii=False,indent=2))
    return len(results)
def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    lock=None
    if args.watch:
        import msvcrt
        lock=(ROOT/'live_report.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    while True:
        count=render()
        if not args.watch or count==10:break
        time.sleep(30)
if __name__=='__main__':main()
