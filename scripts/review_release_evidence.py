"""Read-only release audit; writes evidence, never prints credential values."""
import json
from pathlib import Path
import re
import struct
import subprocess
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    from rescuehandsai.evaluation import HandoffTracker
    report = {}
    tracker = HandoffTracker('fork')
    sequence = [({'right_arm'}, False), ({'right_arm', 'left_arm'}, False),
                ({'left_arm'}, True), ({'left_arm'}, False)]
    states = []
    for holders, supported in sequence:
        tracker.update(SimpleNamespace(held_by={'fork': holders}, supported={'fork': supported}))
        states.append(tracker.stage)
    report['table_interrupted_handoff'] = {'stages': states, 'incorrectly_accepted': tracker.done}
    checkpoint = ROOT / 'models/smolvla_rescuehands'
    config = json.loads((checkpoint / 'config.json').read_text())
    report['checkpoint'] = {'state': config['input_features']['observation.state']['shape'],
                            'action': config['output_features']['action']['shape'],
                            'joint_names_saved': config.get('action_feature_names')}
    stats = {}
    dataset_stats = json.loads((ROOT / 'data/merged_local/meta/stats.json').read_text())
    normalizer_differences = {}
    for path in checkpoint.glob('*.safetensors'):
        with path.open('rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(length))
        tensors = {k: v for k, v in header.items() if k != '__metadata__'}
        if path.name == 'model.safetensors':
            report['checkpoint']['weights'] = {'tensor_count': len(tensors), 'bytes': path.stat().st_size,
                'complete_offsets': max(v['data_offsets'][1] for v in tensors.values()) + 8 + length == path.stat().st_size}
        else:
            stats[path.name] = {k: v['shape'] for k, v in tensors.items()}
            for key, meta in tensors.items():
                feature, stat = key.rsplit('.', 1)
                if feature in ('observation.state', 'action') and stat in ('mean', 'std'):
                    with path.open('rb') as f:
                        f.seek(8 + length + meta['data_offsets'][0])
                        raw = f.read(meta['data_offsets'][1] - meta['data_offsets'][0])
                    if meta['dtype'] != 'F32':
                        raise ValueError(f'Unexpected statistic dtype: {meta["dtype"]}')
                    values = np.frombuffer(raw, dtype='<f4')
                    normalizer_differences[f'{path.name}:{key}'] = {
                        'finite': bool(np.isfinite(values).all()),
                        'max_abs_diff_from_local_dataset': float(np.max(np.abs(values - dataset_stats[feature][stat])))}
    report['normalizer_shapes'] = stats
    report['normalizer_vs_local_dataset'] = normalizer_differences
    summaries = []
    for folder in sorted((ROOT / 'results').glob('scripted_v2_*')):
        summary = json.loads((folder / 'summary.json').read_text())
        episodes = [json.loads(p.read_text()) for p in folder.glob('episode_*.json')]
        actual = sum(e['state'] == 'SUCCEEDED' for e in episodes)
        summaries.append({'run': folder.name, 'episodes': len(episodes), 'counted_successes': actual,
                          'matches_summary': actual == summary['successes'],
                          'policy': summary['policy']['name'],
                          'budget_violations': [e['seed'] for e in episodes if e['steps'] > e['max_steps']]})
    report['saved_result_consistency'] = summaries
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    patterns = {'huggingface_token': r'hf_[A-Za-z0-9]{25,}',
                'github_token': r'(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}',
                'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                'openai_token': r'sk-(?:proj-)?[A-Za-z0-9_-]{40,}',
                'aws_access_id': r'AKIA[A-Z0-9]{16}'}
    findings, scanned = [], 0
    for name in tracked:
        path = ROOT / name
        if not name or not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeError:
            continue
        scanned += 1
        for line_no, line in enumerate(text.splitlines(), 1):
            for label, pattern in patterns.items():
                if re.search(pattern, line):
                    findings.append({'file': name, 'line': line_no, 'type': label})
    report['security'] = {'tracked_text_files_scanned': scanned, 'credential_pattern_matches': findings,
                          'scope': 'Current tracked files under 5 MB; not full history or a dependency vulnerability audit.'}
    output = ROOT / 'artifacts/independent_release_review.json'
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
