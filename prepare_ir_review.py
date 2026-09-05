"""Offline audit only. RGB labels locate broad review regions, never training labels.

Export lossless IR crops and a standalone review page. Every IR annotation starts
unreviewed; model outputs are deliberately hidden from the image annotation UI.
"""
import argparse
import base64
import json
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
from ball_epipolar import EpipolarBallMatcher
from replay_ball_validation import read_model


def png(image):
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError('PNG encoding failed')
    return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')


def build(recording, control, alternative, output):
    root, out = Path(recording), Path(output)
    out.mkdir(parents=True, exist_ok=False)
    labels = json.loads((root/'analysis/rgb_2d_audit/labels.json').read_text())
    runs = [Path(control), Path(alternative)]
    summaries = [json.loads((r/'summary.json').read_text()) for r in runs]
    rows = [{v['frame']: v for v in json.loads((r/'frames.json').read_text())} for r in runs]
    visible = {g['frame'] for g in labels if g['status'] in ('visible', 'partial_visible')}
    hits = [visible - {m['frame'] for m in s['score']['misses']} for s in summaries]
    groups = defaultdict(list)
    for g in labels:
        f = g['frame']
        if f not in visible:
            group = 'rgb_' + g['status']
        elif f in hits[0] and f in hits[1]:
            group = 'both_timely'
        elif f in hits[0]:
            group = 'ncc_only'
        elif f in hits[1]:
            group = 'ffs_only'
        else:
            group = 'both_missed'
        groups[group].append(g)
    selected = []
    for name, values in groups.items():
        count = min(len(values), 24 if name == 'both_missed' else 12)
        selected.extend((name, values[i]) for i in np.unique(np.linspace(0, len(values)-1, count).astype(int)))
    selected.sort(key=lambda x: x[1]['time_s'])
    matcher = EpipolarBallMatcher(read_model(root/'frames'))
    ir_index = []
    for p in sorted((root/'frames').glob('ir_*.npz')):
        with np.load(p) as f:
            ir_index.append((float(f['timestamp_s']), p))
    times = np.array([t for t, _ in ir_index])
    samples = []
    for group, g in selected:
        with np.load(root/'frames'/g['frame']) as f:
            rgb, rgb_t = f['bgr'], float(f['timestamp_s'])
        ir_t, path = ir_index[int(np.argmin(abs(times-rgb_t)))]
        with np.load(path) as f:
            left, right = f['ir_left'], f['ir_right']
            right_t = float(f['right_timestamp_s'])
        uv = g.get('uv')
        if uv is None:
            x0, y0, x1, y1 = 0, 0, left.shape[1], left.shape[0]
        else:
            # Wide region, with a further right-view disparity margin. No
            # inferred ball size/depth or accepted detector point becomes GT.
            a, b = matcher.roi(left, [uv[0]-35, uv[1]-35, 70, 70])
            x0, y0 = np.maximum([0, 0], a-[180, 45]).astype(int)
            x1, y1 = np.minimum(left.shape[::-1], b+[60, 45]).astype(int)
        sample = dict(id=g['frame'], ir_frame=path.name, rgb_t=rgb_t, ir_t=ir_t,
                      right_t=right_t, rgb_status=g['status'], time_s=g['time_s'],
                      origin=[int(x0), int(y0)], shape=[int(y1-y0), int(x1-x0)],
                      left=dict(status='unreviewed', uv=None), right=dict(status='unreviewed', uv=None),
                      review_note='')
        context = cv2.resize(rgb, (640, 360))
        if uv is not None:
            cv2.drawMarker(context, tuple(np.rint(np.array(uv)/2).astype(int)), (0, 255, 0),
                           cv2.MARKER_CROSS, 14, 1)
        media = dict(rgb=png(context))
        for side, image in [('left', left), ('right', right)]:
            crop = image[y0:y1, x0:x1]
            cv2.imwrite(str(out/(g['frame']+'_'+side+'.png')), crop)
            media[side] = png(crop)
            lo, hi = np.percentile(crop, [1, 99])
            media[side+'_contrast'] = png(np.clip((crop.astype(float)-lo)*255/max(1., hi-lo), 0, 255).astype('uint8'))
        sample['media'] = media
        samples.append(sample)
    diagnostics = {}
    for name, lookup, successful in zip(('ncc', 'ffs'), rows, hits):
        missed = [lookup.get(f, {}) for f in sorted(visible-successful)]
        reasons = Counter(r.get('reason', 'dropped') for r in missed)
        attempt_reasons = Counter(a.get('reason', 'unknown') for r in missed for a in r.get('attempts', []))
        rejection_counts = Counter()
        for r in missed:
            for a in r.get('attempts', []):
                rejection_counts.update(a.get('rejected', {}))
        diagnostics[name] = dict(timely=len(successful), misses=len(missed), frame_reasons=dict(reasons),
                                 attempt_reasons=dict(attempt_reasons), candidate_rejections=dict(rejection_counts))
    audit = dict(visible=len(visible), diagnostics=diagnostics, union_timely=len(hits[0]|hits[1]),
                 ffs_unique=sorted(hits[1]-hits[0]), groups={k: len(v) for k,v in groups.items()},
                 review_samples=len(samples), all_ir_labels_unreviewed=True,
                 scope='Development audit; nearest IR selected offline, not a causal detector evaluation. '
                       'RGB visibility does not establish IR visibility. No metric 3D ground truth. '
                       'Group strata and repeated-video samples are not an independent test set.')
    (out/'diagnostics.json').write_text(json.dumps(audit, indent=2))
    (out/'annotations.json').write_text(json.dumps([{k:v for k,v in s.items() if k!='media'} for s in samples], indent=2))
    template = Path(__file__).with_name('ir_review_template.html').read_text(encoding='utf-8')
    (out/'review.html').write_text(template.replace('__SAMPLES__', json.dumps(samples)), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('recording', 'control', 'alternative', 'output'):
        p.add_argument(name)
    a = p.parse_args()
    build(a.recording, a.control, a.alternative, a.output)
