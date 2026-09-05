"""Replay actual arrival schedules; bounded queues drop frames under load.

Decoding happens before the clock starts, as camera inputs are already arrays.
No future images or labels are provided to the detector. Labels are read only
after workers terminate. Device arrival metadata is replayed, with negative
global-clock offsets clamped to zero; absolute hardware timing remains estimated.
"""
import argparse
from collections import Counter,deque
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
import cv2
import numpy as np
from realtime_source import LatestQueue,RealtimeSource,load_recording,percentiles
from replay_ball_validation import make_system,read_model
import d455_table_tennis_tracker as app


def score_outputs(root,events,rows,deadline_ms):
    meta=json.loads((root/'camera.json').read_text());start=meta['action_start_sensor_timestamp_s']
    labels=json.loads((root/'analysis/rgb_2d_audit/labels.json').read_text())
    rgb=sorted((e for e in events if e['kind']=='rgb'),key=lambda e:e['t']);ts=np.array([e['t'] for e in rgb])
    matched={};false=[]
    for row in rows:
        if not row.get('measured') or row.get('xyz') is None:continue
        # RGB visibility labels belong to the RGB identity observation, not to
        # an IR point captured at a neighbouring time. This is coverage only.
        identity_t=row.get('rgb_timestamp_s',row['measurement_t'])
        j=int(np.argmin(abs(ts-identity_t)));name=rgb[j]['name']
        if abs(ts[j]-identity_t)>.012:continue
        matched.setdefault(name,[]).append(row)
    visible=hit=deadline_hit=uncertain=0;misses=[]
    for g in labels:
        if g['status']=='uncertain_visibility':uncertain+=1;continue
        seen=g['status'] in ('visible','partial_visible');visible+=seen
        candidates=matched.get(g['frame'],[])
        correct=[r for r in candidates if seen and r.get('rgb_uv') is not None and
                 np.linalg.norm(np.array(r['rgb_uv'])-g['uv'])<=g['tolerance_px']]
        timely=[r for r in correct if r['latency_ms']<=deadline_ms]
        hit+=bool(correct);deadline_hit+=bool(timely)
        if seen and not timely:misses.append(dict(frame=g['frame'],status=g['status'],has_measurement=bool(candidates)))
        if candidates and not correct:false.append(g['frame'])
    formal=[r for r in rows if start<=r['measurement_t']<start+meta['requested_duration_s']]
    return dict(visible_rgb_frames=visible,matched_measured_3d_frames=hit,
        measured_3d_frame_recall=hit/max(1,visible),deadline_ms=deadline_ms,
        within_deadline_3d_frames=deadline_hit,within_deadline_3d_recall=deadline_hit/max(1,visible),
        uncertain_frames=uncertain,frames_with_off_target_measurements=false,misses=misses,
        formal_measurement_updates=sum(r.get('measured',False) for r in formal),
        measurement_latency_ms=percentiles([r['latency_ms'] for r in formal if r.get('measured')]),
        scope='30 Hz RGB visibility denominator. Identity observations are assigned to their RGB timestamp '
              '(or nearest frame within 12 ms for legacy output); RGB location must be within 10 px. '
              'IR XYZ uses its own capture timestamp, separate from RGB identity. '
              'This tests measured-output availability and RGB identity, not correctness '
              'of the IR correspondence or metric 3D accuracy. Predictions do not count; no 3D ground truth.')


def run(args):
    root=Path(args.recording);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    print('[LOAD] decoding before real-time clock starts',flush=True)
    events=load_recording(root);source=RealtimeSource(events)
    s=make_system(read_model(root/'frames'),rgb_detector='image')
    rgbq=LatestQueue();irq=LatestQueue();rows=[];rgb_rows=[];errors=[];timings={'rgb':[],'ir':[]}
    def rgb_worker():
        while not source.done.is_set() or not rgbq.q.empty():
            try:e=rgbq.get()
            except queue.Empty:continue
            t0=time.perf_counter()
            try:
                s._process_rgb_frame(e['bgr'],e['depth'],e['t'],e['depth_t'])
                result=s.rgb_state.get_image_observation()
                rgb_rows.append(dict(frame=e['name'],result=result,latency_ms=(time.perf_counter()-source.capture_wall(e['t']))*1000))
                s.table_queue.put_latest((e['bgr'],e['depth'],e['t']))
            except Exception as ex:errors.append(repr(ex));source.cancel.set()
            timings['rgb'].append(1000*(time.perf_counter()-t0))
    def ir_worker():
        while not source.done.is_set() or not irq.q.empty():
            try:e=irq.get()
            except queue.Empty:continue
            t0=time.perf_counter()
            try:
                packet=SimpleNamespace(timestamp_s=e['t'],host_timestamp_s=time.monotonic(),
                    frame_number=e['frame_number'],ir_left=e['ir_left'],ir_right=e['ir_right'])
                table,meta=s._table_snapshot_left(e['t']);s._sync_table_frame(table)
                measured,kind,measurement,_=s._process_ball_packet(packet,table)
                payload=s._publish(packet,table,measured,kind,measurement if measured else None,meta)
                json.dumps(payload)  # Account for serialization; disk writes occur after the timed run.
                b=payload['ball'];mt=b['measurement_timestamp_s'] if measured else e['t']
                xyz=b.get('last_measurement_camera_m') if measured else b.get('position_camera_m')
                uv=None if xyz is None else app.project_rs(s.model.color_intr,app.transform_point(s.model.T_color_left,np.array(xyz)))
                rows.append(dict(frame=e['name'],measurement_t=mt,measured=measured,valid=b['valid'],
                    source=kind,xyz=xyz,rgb_uv=None if uv is None else uv.tolist(),state=b['state'],
                    latency_ms=(time.perf_counter()-source.capture_wall(mt))*1000,
                    output_t=e['t'],rejects=s.selector.diagnostics.copy()))
            except Exception as ex:errors.append(repr(ex));source.cancel.set()
            timings['ir'].append(1000*(time.perf_counter()-t0))
    workers=[threading.Thread(target=rgb_worker),threading.Thread(target=ir_worker)]
    table_thread=threading.Thread(target=s._table_worker);table_thread.start()
    for w in workers:w.start()
    print('[RUN] 1x wall-clock arrival replay, queue capacity one',flush=True)
    source.run(lambda e:(rgbq if e['kind']=='rgb' else irq).put(e))
    for w in workers:w.join(timeout=5)
    s.stop_event.set();table_thread.join(timeout=5)
    if any(w.is_alive() for w in workers):errors.append('worker_did_not_stop')
    score=score_outputs(root,events,rows,args.deadline_ms)
    summary=dict(score=score,delivered=dict(source.delivered),processed=dict(rgb=len(rgb_rows),ir=len(rows)),
        dropped=dict(rgb=rgbq.dropped,ir=irq.dropped,table=s.table_queue.dropped),
        scheduler_lateness_ms=percentiles(source.jitter_ms),compute_ms={k:percentiles(v) for k,v in timings.items()},
        errors=errors,source_span_s=max(e['available_t'] for e in events)-source.first,
        replay_wall_s=time.perf_counter()-source.start_wall,
        main_sha256=hashlib.sha256(Path(app.__file__).read_bytes()).hexdigest(),
        timing_note='Recorded SDK USB arrival metadata; negative global-clock offsets clamped to zero. '
                    'Predecoded/aligned inputs; SDK alignment cost, disk recording and display are not timed.')
    (out/'frames.json').write_text(json.dumps(rows,indent=2));(out/'rgb_frames.json').write_text(json.dumps(rgb_rows,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='score'},indent=2))
    print(json.dumps({k:v for k,v in score.items() if k not in ('misses','frames_with_off_target_measurements')},indent=2))
    if errors:raise RuntimeError(errors)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('recording');p.add_argument('output');p.add_argument('--deadline-ms',type=float,default=50.)
    run(p.parse_args())
