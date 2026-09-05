"""Causal wall-clock replay for the experimental low-latency RGB/stereo path."""
import argparse,json,hashlib,threading,time,queue
from collections import deque,Counter
from pathlib import Path
import cv2
from fast_rgbd import FastRGBDProcessor
from realtime_source import load_recording,RealtimeSource,LatestQueue,percentiles
from replay_realtime_3d import score_outputs
from replay_ball_validation import read_model


def run(args):
    root=Path(args.recording);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    print('[LOAD] predecode arrays before timed run',flush=True);events=load_recording(root)
    cv2.setNumThreads(1);source=RealtimeSource(events);model=read_model(root/'frames')
    stereo=None
    if args.stereo_backend=='ffs':
        from ball_foundation import FoundationBallMatcher
        stereo=FoundationBallMatcher(model)
        print('[FFS] batch-one warmup latency ms: '+json.dumps(stereo.warmup_stats),flush=True)
    processor=FastRGBDProcessor(model,stereo=stereo)
    lock=threading.Lock();ir=deque(maxlen=24);rgbq=LatestQueue();rows=[];errors=[]
    ready=threading.Event();worker_warmup_ms=[]
    def feed(e):
        if e['kind']=='rgb':rgbq.put(e)
        else:
            with lock:ir.append(e)
    def worker():
        try:
            if stereo is not None:
                begin=time.perf_counter()
                # CUDA/cuDNN handles may initialize per thread. Warm the
                # actual processing thread before announcing that input is ready.
                stereo.warmup_stats=stereo.runtime.warmup(stereo.width,stereo.height)
                stereo.minimum_ms=stereo.warmup_stats['p95']+2.
                worker_warmup_ms.append(1000*(time.perf_counter()-begin))
        except Exception as ex:
            errors.append(repr(ex));ready.set();return
        ready.set()
        while not source.done.is_set() or not rgbq.q.empty():
            try:e=rgbq.get()
            except queue.Empty:continue
            try:
                begin=time.perf_counter()
                # Snapshot only frames already delivered by the real-time source.
                # More frames may arrive while RGB is computed, but this version
                # deliberately never reads future/offline array indices.
                def available():
                    with lock:return list(ir)
                deadline=source.capture_wall(e['t'])+(args.deadline_ms-2.)/1000.
                row=processor.process(e['bgr'],e['t'],available,deadline)
                row.update(frame=e['name'],latency_ms=1000*(time.perf_counter()-source.capture_wall(e['t'])),
                    compute_ms=1000*(time.perf_counter()-begin))
                json.dumps(row)
                now=time.perf_counter()
                row['rgb_latency_ms']=1000*(now-source.capture_wall(e['t']))
                oldest=min(e['t'],row['stereo_t']) if row['measured'] else e['t']
                row['latency_ms']=1000*(now-source.capture_wall(oldest))
                row['stereo_latency_ms']=None if not row['measured'] else 1000*(now-source.capture_wall(row['stereo_t']))
                rows.append(row)
            except Exception as ex:errors.append(repr(ex));source.cancel.set()
    thread=threading.Thread(target=worker);thread.start()
    if not ready.wait(timeout=60) or errors:
        source.cancel.set();source.done.set();thread.join(timeout=5)
        raise RuntimeError(errors or ['worker_initialization_timeout'])
    print('[RUN] original USB arrival schedule, one-slot RGB queue',flush=True)
    source.run(feed);thread.join(timeout=5)
    if thread.is_alive():errors.append('worker_stalled')
    score=score_outputs(root,events,rows,args.deadline_ms)
    summary=dict(score=score,stereo_backend=args.stereo_backend,worker_startup_warmup_ms=worker_warmup_ms,
        foundation_stats=None if stereo is None else dict(warmup_ms=stereo.warmup_stats,inference_calls=stereo.calls,
            budget_skips=stereo.budget_skips,checkpoint_sha256=stereo.runtime.sha256),
        delivered=dict(source.delivered),processed_rgb=len(rows),dropped_rgb=rgbq.dropped,
        provisional_95_deadline_target_met=score['within_deadline_3d_recall']>=.95 and not errors,
        compute_ms=percentiles([r['compute_ms'] for r in rows]),image_ms=percentiles([r['image_ms'] for r in rows]),
        stereo_ms=percentiles([r.get('stereo_ms',0) for r in rows]),
        all_rgb_latency_ms=percentiles([r['latency_ms'] for r in rows]),
        scheduler_lateness_ms=percentiles(source.jitter_ms),image_modes=dict(Counter(r['image_mode'] for r in rows)),
        source_span_s=max(e['available_t'] for e in events)-source.first,
        replay_wall_s=time.perf_counter()-source.start_wall,errors=errors,
        hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in
                ('ball_image_roi.py','ball_image.py','ball_epipolar.py','fast_rgbd.py')},
        scope='Experimental camera-frame ball measurement only, no table pose/filter/display/recording cost. '
              'Predecoded arrays; complete XYZ triangulated from one synchronized IR pair. '
              'RGB identity evidence has a separate timestamp within 18 ms. '
              'Each stereo pair is counted once. Latency is measured from the older of RGB and stereo capture. '
              'Causal host-arrival replay; no predictions count as detections. No 3D ground truth.')
    (out/'frames.json').write_text(json.dumps(rows,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2))
    compact=dict(summary);compact['score']={k:v for k,v in score.items() if k not in ('misses','frames_with_off_target_measurements')}
    print(json.dumps(compact,indent=2))
    if errors:raise RuntimeError(errors)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('recording');p.add_argument('output');p.add_argument('--deadline-ms',type=float,default=50.)
    p.add_argument('--stereo-backend',choices=('ncc','ffs'),default='ncc')
    run(p.parse_args())
