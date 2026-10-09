"""Single-instance, fail-closed local experiment watchdog (no GPU/model imports)."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import psutil

FATAL = ('out of memory', 'outofmemoryerror', 'cuda error', 'device-side assert',
         'hash mismatch', 'frozen file drift', 'parity', 'schema failure',
         'invalid schema', 'provenance changed', 'foreign-model', 'conflicting context')
CHILDREN = []


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def verify_receipt(path, seen=None):
    seen = {} if seen is None else seen
    value = read(path)
    if not isinstance(value.get('files'), dict) or not value['files']:
        raise ValueError('Invalid/empty freeze receipt: ' + str(path))
    for name, expected in value['files'].items():
        if name in seen:
            if seen[name] != expected:
                raise ValueError('Conflicting freeze hashes')
            continue
        with Path(name).open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError('Frozen file drift: ' + name)
        seen[name] = actual
        if Path(name).suffix == '.json':
            child = read(name)
            if isinstance(child, dict) and isinstance(child.get('files'), dict):
                verify_receipt(name, seen)


def identity(pid):
    p = psutil.Process(pid)
    return dict(pid=pid, created=p.create_time(), command=p.cmdline())


def alive(saved):
    if not saved:
        return False
    try:
        current = identity(saved['pid'])
        return current == saved and psutil.Process(saved['pid']).is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def other_python():
    # A Windows venv launcher and its runtime are both allowed for this watchdog.
    own = psutil.Process()
    excluded = {own.pid, *(p.pid for p in own.parents())}
    found = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        if p.pid not in excluded and (p.info['name'] or '').lower() in ('python.exe', 'pythonw.exe', 'python', 'python3'):
            found.append(p.pid)
    return found


def held(root):
    return any((root / x).exists() for x in ('PAUSED_BY_USER.json', 'DIAGNOSTIC_HOLD.json', 'WATCHDOG_STOP'))


def fatal_log(path):
    if not path or not Path(path).exists():
        return None
    with Path(path).open('rb') as f:
        f.seek(max(0, f.seek(0, 2) - 131072))
        text = f.read().decode('utf8', errors='replace').lower()
    return next((word for word in FATAL if word in text), None)


def retry_delay(attempts):
    return min(60 * (2 ** max(0, attempts - 1)), 900)


def tick(root, config, state):
    CHILDREN[:] = [p for p in CHILDREN if p.poll() is None]
    now = time.time()
    if held(root):
        return 'paused_by_hold'
    queue = read(config['queue'])
    if queue.get('status') in ('PAUSED_BY_USER', 'AWAITING_CARS_DUPLICATE_POLICY', 'BLOCKED'):
        return 'paused_by_queue'
    if not queue.get('execution_ready') or not config.get('command'):
        return 'waiting_for_verified_runner'
    if state.get('blocked'):
        return 'blocked_requires_review'
    binding = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if state.get('binding') not in (None, binding):
        raise ValueError('Watchdog launch configuration changed; review required')
    state['binding'] = binding
    if alive(state.get('process')):
        return 'running'
    if other_python():
        return 'waiting_for_other_python'
    # The registration must bind both this exact config and the resume verifier.
    receipt = Path(config['freeze'])
    files = read(receipt)['files']
    for field, command_field in (('runner_script', 'command'), ('verifier_script', 'verify_command')):
        if str(Path(config[field]).resolve()) not in [str(Path(x).resolve()) for x in config[command_field][1:] if not x.startswith('-')]:
            raise ValueError('Registered script missing from command: ' + field)
    for required in (str((root / 'watchdog_config.json').resolve()), str(Path(config['verifier_script']).resolve()), str(Path(config['runner_script']).resolve()), str(Path(__file__).resolve())):
        if required not in files:
            raise ValueError('Registration freeze does not bind required file: ' + required)
    verify_receipt(receipt)
    complete = Path(config['complete'])
    if complete.exists():
        verify_receipt(complete)
        return 'complete'
    reason = fatal_log(state.get('stderr'))
    stage_status = Path(config.get('stage_status', root / 'status.json'))
    if stage_status.exists():
        status_value = read(stage_status)
        error_text = str(status_value.get('error', '')).lower()
        reason = reason or next((word for word in FATAL if word in error_text), None)
        reason = reason or fatal_log(status_value.get('stderr'))
    if reason:
        raise ValueError('Non-retryable failure in latest log: ' + reason)
    if state.get('process') and not state.get('exit_observed'):
        state['exit_observed'] = now
        return 'exit_detected_backoff'
    attempts = state.get('attempts', [])
    recent = [t for t in attempts if now - t < 21600]
    if len(recent) >= 4:
        raise ValueError('Four launches within six hours; restart loop stopped')
    if state.get('exit_observed') and now - state['exit_observed'] < retry_delay(len(recent)):
        return 'retry_backoff'
    # This runner-specific, CPU-only command must verify all current stage freezes,
    # exact cache keys, manifest hashes, batch/contract parity and resume eligibility.
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    stamp = time.strftime('%Y%m%d_%H%M%S')
    validation_log = root / ('watchdog_verify_' + stamp + '.log')
    with validation_log.open('w', encoding='utf8') as f:
        result = subprocess.run(config['verify_command'], cwd=config['cwd'], stdout=f,
                                stderr=subprocess.STDOUT, creationflags=flags, timeout=900)
    if result.returncode:
        raise ValueError('Resume validation failed; see ' + str(validation_log))
    verify_receipt(receipt)
    # Check again after validation; the user may have paused during hashing.
    if held(root) or not read(config['queue']).get('execution_ready'):
        return 'paused_after_validation'
    if other_python():
        return 'waiting_for_other_python'
    stdout, stderr = root / ('watchdog_run_' + stamp + '.stdout.log'), root / ('watchdog_run_' + stamp + '.stderr.log')
    with stdout.open('w', encoding='utf8') as fo, stderr.open('w', encoding='utf8') as fe:
        p = subprocess.Popen(config['command'], cwd=config['cwd'], stdout=fo, stderr=fe,
                             creationflags=flags, env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'})
    CHILDREN.append(p)
    state.update(process=identity(p.pid), stdout=str(stdout), stderr=str(stderr),
                 attempts=recent + [now], exit_observed=None)
    return 'launched'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    # Kernel lock is released on process exit, including crashes/reboots.
    with (root / 'watchdog.lock').open('a+b') as lock:
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                if lock.read(1) == b'':
                    lock.write(b'0'); lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise SystemExit('A watchdog already holds this queue lock')
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / 'watchdog_state.json'
        state = read(state_path) if state_path.exists() else {}
        while True:
            config = read(root / 'watchdog_config.json')
            try:
                status = tick(root, config, state)
            except Exception as error:
                state['blocked'] = str(error)
                status = 'blocked_requires_review'
            state.update(status=status, checked_at=time.time(), watchdog=identity(os.getpid()))
            write(state_path, state)
            if args.once or status in ('complete', 'paused_by_hold', 'paused_by_queue', 'blocked_requires_review'):
                return
            time.sleep(max(5, int(config.get('interval_seconds', 30))))


if __name__ == '__main__':
    main()
