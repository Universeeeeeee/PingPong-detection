"""Replay timestamped RGBD + stereo without opening a physical camera.

Manual pixel annotations used below are limited to this recorded static scene;
they are not an accuracy claim for flight, occlusion, or unseen environments.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import Mock

import cv2
import numpy as np
import pyrealsense2 as rs
import d455_table_tennis_tracker as app


def read_model(directory):
    m=json.loads((Path(directory)/'camera.json').read_text())
    for name in ('ir_left_intr','ir_right_intr','color_intr'):
        data=m[name];intr=rs.intrinsics()
        for k in ('width','height','fx','fy','ppx','ppy','coeffs'):setattr(intr,k,data[k])
        intr.model=getattr(rs.distortion,data['model'].split('.')[-1]);m[name]=intr
    for name in ('T_left_right','T_right_left','T_left_color','T_color_left'):m[name]=np.array(m[name])
    return SimpleNamespace(**m)


def make_system(model,output=None,rgb_detector='hsv'):
    args=app.build_arg_parser().parse_args([])
    args.zmq_port=0;args.table_min_area=15000
    args.table_hsv_low=np.array([85,80,40],np.uint8);args.table_hsv_high=np.array([135,255,255],np.uint8)
    args.ball_video=None if output is None else str(Path(output)/'ball_overlay.mp4')
    s=app.TableTennisPerceptionSystem(args)
    s.pub_socket.close(linger=0);s.pub_socket=Mock()
    s.model=model
    s.args.rgb_detector=rgb_detector
    s._initialize_rgb_detector()
    s.left_ir_detector=app.IRBallDetector(model.ir_left_intr,app.IRDetectorConfig(),'left')
    s.right_ir_detector=app.IRBallDetector(model.ir_right_intr,app.IRDetectorConfig(),'right')
    s.selector=app.StereoCandidateSelector(model,args.ball_radius,s.ball_validation)
    s.table_tracker=app.TablePoseTracker(color_intr=model.color_intr,depth_scale=model.depth_scale,
        table_length_m=2.74,table_width_m=1.525,hsv_ranges=app.table_hsv_ranges(args),
        pose_file=Path('/tmp/unused_ball_replay_table_pose.npz'),min_area=15000,
        confirm_frames=5,hold_seconds=.6,edge_tolerance_px=5.,validation_hz=12.)
    return s


def replay(directory,output,gt=None,limit=None,rgb_delay=0.,occlude_rgb=None,rgb_detector='auto'):
    directory,output=Path(directory),Path(output);output.mkdir(parents=True,exist_ok=True)
    s=make_system(read_model(directory),output,rgb_detector)
    events=[]
    for kind in ('rgb','ir'):
        for path in sorted(directory.glob(kind+'_*.npz')):
            with np.load(path) as f:events.append((float(f['timestamp_s']),kind,path))
    first_time=min(e[0] for e in events)
    events.sort(key=lambda e:e[0]+(rgb_delay if e[1]=='rgb' else 0.))
    if limit is not None:events=events[:limit]
    counters=Counter();rejects=Counter();rows=[];rgb_rows=[];timings={'rgb':[],'ir':[]}
    previous=None
    try:
        for t,kind,path in events:
            f=np.load(path);start=time.perf_counter()
            if kind=='rgb':
                bgr=f['bgr'].copy();depth=f['depth'].copy()
                if occlude_rgb is not None and gt is not None and occlude_rgb[0]<=t-first_time<=occlude_rgb[1]:
                    x,y=map(int,gt);x0,x1=max(0,x-20),min(bgr.shape[1],x+21);y0,y1=max(0,y-20),min(bgr.shape[0],y+21)
                    bgr[y0:y1,x0:x1]=bgr[y,min(bgr.shape[1]-1,x+28)]
                    depth[y0:y1,x0:x1]=0
                s.table_tracker.update(bgr,depth,t)
                cs=s._process_rgb_frame(bgr,depth,t,float(f['depth_timestamp_s']))
                accepted=[c for c in cs if c.validation['identity_ok']]
                near=[c for c in accepted if gt is not None and np.linalg.norm(c.center-gt)<15.]
                counters['rgb_frames']+=1;counters['true_rgb_candidates']+=bool(near)
                rejects.update(c.validation['rejection_reason'] for c in cs if c.validation['rejection_reason'])
                rgb_rows.append(dict(frame=path.name,result=s.rgb_state.get_image_observation(),candidates=[dict(uv=c.center.tolist(),score=c.score,**c.validation) for c in cs]))
            else:
                packet=SimpleNamespace(timestamp_s=t,host_timestamp_s=t,frame_number=int(f['frame_number']),
                                       ir_left=f['ir_left'],ir_right=f['ir_right'])
                table,metadata=s._table_snapshot_left(t);s._sync_table_frame(table)
                measured,source,measurement,_=s._process_ball_packet(packet,table)
                payload=s._publish(packet,table,measured,source,measurement if measured else None,metadata)
                b=payload['ball'];counters['ir_frames']+=1;counters[b['state']]+=1;counters[source]+=measured
                uv=None
                if b['valid']:
                    uv=app.project_rs(s.model.color_intr,app.transform_point(s.model.T_color_left,np.array(b['position_camera_m'])))
                    if gt is not None:
                        counters['on_target_output_frames' if uv is not None and np.linalg.norm(uv-gt)<15. else 'off_target_output_frames']+=1
                if measured and gt is not None:
                    raw=b['last_measurement_camera_m']
                    raw_uv=app.project_rs(s.model.color_intr,app.transform_point(s.model.T_color_left,np.array(raw)))
                    counters['true_measurements' if raw_uv is not None and np.linalg.norm(raw_uv-gt)<15. else 'nonball_measurements']+=1
                if b['track_id']!=previous and b['valid']:
                    counters['track_starts']+=1;previous=b['track_id']
                rows.append(dict(timestamp_s=t,frame=path.name,ball=b,ball_2d=payload['ball_2d'],
                    projected_rgb_uv=None if uv is None else uv.tolist(),diagnostics=payload['diagnostics']))
            timings[kind].append((time.perf_counter()-start)*1000.)
        cv2.imwrite(str(output/'ball_overlay.jpg'),s.rgb_state.get_debug())
    finally:
        if s.ball_video_writer is not None:s.ball_video_writer.release()
    summary=dict(counts=dict(counters),rgb_rejection_reasons=dict(rejects),
                 processing_ms={k:dict(median=float(np.median(v)),p95=float(np.percentile(v,95))) for k,v in timings.items()},
                 manual_static_ball_uv=gt,scope='This timestamped static scene only; image candidate recall is not 3D tracking recall.')
    summary.update(simulated_rgb_delay_seconds=rgb_delay,synthetic_rgb_occlusion_seconds=occlude_rgb,rgb_detector=rgb_detector)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'frames.json').write_text(json.dumps(rows,indent=2))
    (output/'rgb_candidates.json').write_text(json.dumps(rgb_rows,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('output')
    p.add_argument('--static-ball-uv',type=float,nargs=2);p.add_argument('--limit',type=int)
    p.add_argument('--rgb-delay-seconds',type=float,default=0.)
    p.add_argument('--occlude-rgb',type=float,nargs=2)
    p.add_argument('--rgb-detector',choices=['auto','image','hsv'],default='auto')
    a=p.parse_args();replay(a.directory,a.output,a.static_ball_uv,a.limit,a.rgb_delay_seconds,a.occlude_rgb,a.rgb_detector)
