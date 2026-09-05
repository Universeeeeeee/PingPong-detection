"""Verify native SDK playback, or export timestamped arrays for offline replay."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pyrealsense2 as rs


def export(directory, verify_only=False, start_seconds=0., duration=None):
    directory=Path(directory)
    meta=json.loads((directory/'camera.json').read_text())
    output=directory/'frames'
    if not verify_only:output.mkdir(exist_ok=False)
    cfg=rs.config();cfg.enable_device_from_file(str(directory/'capture.db3'),repeat_playback=False)
    cfg.enable_all_streams();pipe=rs.pipeline();profile=pipe.start(cfg)
    playback=profile.get_device().as_playback();playback.set_real_time(False)
    align=rs.align(rs.stream.color)
    counts={};seen={};gaps={};first=None;rgb_n=ir_n=0;idle=time.monotonic();error=None
    try:
        while True:
            ok,frames=pipe.try_wait_for_frames(500)
            if not ok:
                if playback.current_status()==rs.playback_status.stopped:break
                if time.monotonic()-idle>10:raise RuntimeError('Playback stalled')
                continue
            idle=time.monotonic()
            fresh=set()
            for f in frames:
                p=f.get_profile();name=str(p.stream_type()).split('.')[-1]+'_'+str(p.stream_index())
                n=int(f.get_frame_number());previous=seen.get(name)
                if previous is not None and n<=previous:continue
                counts[name]=counts.get(name,0)+1;seen[name]=n;fresh.add(name)
                gaps[name]=gaps.get(name,0)+(max(0,n-previous-1) if previous is not None else 0)
            left,right=frames.get_infrared_frame(1),frames.get_infrared_frame(2)
            depth,color=frames.get_depth_frame(),frames.get_color_frame()
            if not left:continue
            t=left.get_timestamp()*.001
            if first is None:first=t
            elapsed=t-first
            if elapsed<start_seconds:continue
            if duration is not None and elapsed>=start_seconds+duration:break
            if verify_only:continue
            if left and right and {'infrared_1','infrared_2'}<=fresh:
                np.savez_compressed(output/('ir_%04d.npz'%ir_n),
                    ir_left=np.asanyarray(left.get_data()),ir_right=np.asanyarray(right.get_data()),
                    depth_raw=np.asanyarray(depth.get_data()) if depth else np.empty((0,0),np.uint16),
                    timestamp_s=t,right_timestamp_s=right.get_timestamp()*.001,
                    frame_number=left.get_frame_number(),right_frame_number=right.get_frame_number())
                ir_n+=1
            if color and depth and 'color_0' in fresh:
                aligned=align.process(frames);c,d=aligned.get_color_frame(),aligned.get_depth_frame()
                np.savez_compressed(output/('rgb_%04d.npz'%rgb_n),bgr=np.asanyarray(c.get_data()),
                    depth=np.asanyarray(d.get_data()),timestamp_s=c.get_timestamp()*.001,
                    depth_timestamp_s=d.get_timestamp()*.001,frame_number=c.get_frame_number())
                rgb_n+=1
        if not all(counts.get(k,0) for k in ('color_0','depth_0','infrared_1','infrared_2')):
            raise RuntimeError('Missing required stream in SDK playback')
    except Exception as exc:
        error=str(exc);raise
    finally:
        pipe.stop()
        report=dict(playback_counts=counts,playback_frame_gaps=gaps,error=error,
                    exported_rgb_frames=rgb_n,exported_ir_frames=ir_n,verify_only=verify_only,
                    start_seconds=start_seconds,duration=duration,
                    note='Counts reflect SDK synchronized playback; compare with callback diagnostics for loss.')
        (directory/('playback_check.json' if verify_only else 'export_report.json')).write_text(json.dumps(report,indent=2)+'\n')
        if not verify_only:
            meta.update(rgb_frames=rgb_n,ir_frames=ir_n,export_error=error)
            (output/'camera.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory')
    p.add_argument('--verify-only',action='store_true');p.add_argument('--start-seconds',type=float,default=0.)
    p.add_argument('--duration',type=float)
    a=p.parse_args()
    if a.start_seconds<0 or (a.duration is not None and a.duration<=0):p.error('Invalid time range')
    export(a.directory,a.verify_only,a.start_seconds,a.duration)
