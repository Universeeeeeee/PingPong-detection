#!/usr/bin/env python3
"""Replay the saved RGB-D commissioning frames without opening a camera."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,pyrealsense2 as rs
from table_pose_geometry import FixedTablePoseTracker


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,default=Path('replay_results'))
    p.add_argument('--compare-legacy',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    m=json.loads((args.input/'intrinsics.json').read_text());intr=rs.intrinsics()
    for k in ('width','height','fx','fy','ppx','ppy','coeffs'):setattr(intr,k,m[k])
    intr.model=getattr(rs.distortion,m['model'].split('.')[-1])
    t=FixedTablePoseTracker(intr,m['depth_scale'],2.74,1.525,[(np.array([85,80,40],np.uint8),np.array([135,255,255],np.uint8))],args.output/'pose.npz')
    legacy=None
    if args.compare_legacy:
        from d455_table_tennis_tracker import LegacyTablePoseTracker
        legacy=LegacyTablePoseTracker(intr,m['depth_scale'],2.74,1.525,t.hsv_ranges,args.output/'legacy_pose.npz',15000)
    records=[]
    video=None;video_start=None;video_frames=0
    for path in sorted(args.input.glob('frame_*.npz')):
        f=np.load(path);start=time.perf_counter();ts=float(f['timestamp_s']);t.update(f['bgr'],f['depth'],ts)
        snap,meta=t.snapshot(ts);meta['processing_ms']=1000*(time.perf_counter()-start)
        meta['frame']=path.name;meta['T_camera_table']=None if snap is None else snap.T.tolist();records.append(meta)
        image=t.get_debug_image()
        if image is not None:cv2.imwrite(str(args.output/(path.stem+'.jpg')),image)
        if legacy is not None and image is not None:
            try:
                legacy.update(f['bgr'],f['depth'],ts)
                old=legacy.get_debug_image()
                old_pose=legacy.predict(ts)
                meta['legacy_T_camera_table']=None if old_pose is None else old_pose.T.tolist()
            except Exception as exc:
                old=f['bgr'].copy();meta['legacy_error']=str(exc)
            if old is None:old=f['bgr'].copy()
            comparison=np.hstack((old,image))
            cv2.putText(comparison,'BEFORE',(15,110),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(comparison,'AFTER',(intr.width+15,110),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.imwrite(str(args.output/'comparison.jpg'),comparison)
            if video is None:
                video=cv2.VideoWriter(str(args.output/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15.,(intr.width*2,intr.height));video_start=ts
                if not video.isOpened():raise RuntimeError('Could not open comparison video')
            target=int((ts-video_start)*15)+1
            while video_frames<target:video.write(comparison);video_frames+=1
    if video is not None:video.release()
    (args.output/'results.json').write_text(json.dumps(records,indent=2))
    print(json.dumps({'frames':len(records),'valid':sum(r['valid'] for r in records),'states':[r['state'] for r in records],'mean_ms':float(np.mean([r['processing_ms'] for r in records]))},indent=2))


if __name__=='__main__':main()
