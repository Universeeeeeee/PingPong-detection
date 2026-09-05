"""Fast RGB-only D455 entry point. No depth is fabricated or required."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ball_image import ImageBallDetector


def run(args):
    out=Path(args.output) if args.output else Path(__file__).resolve().parent/'runs'/'rgb_2d'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=True)
    ranges=[(np.array([2,75,65],np.uint8),np.array([35,255,255],np.uint8))]
    detector=ImageBallDetector(ranges,orange=True)
    pipe=rs.pipeline();cfg=rs.config()
    if args.bag:cfg.enable_device_from_file(str(Path(args.bag).resolve()),repeat_playback=False)
    elif args.serial:cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color,1280,720,rs.format.bgr8,30)
    stop=False;started=False;writer=None;rows=None;frame_count=0;frame_gaps=0;last_number=None;timings=[]
    first_t=None;last_t=None;status='failed';error=None;serial=None;duplicates=0;playback=None
    def interrupt(*_):
        nonlocal stop
        stop=True
    old_handler=signal.signal(signal.SIGINT,interrupt)
    print(f'[INFO] RGB 2D only; output: {out}',flush=True)
    try:
        profile=pipe.start(cfg);started=True
        serial=profile.get_device().get_info(rs.camera_info.serial_number)
        if args.bag:
            playback=profile.get_device().as_playback();playback.set_real_time(False)
        rows=(out/'detections.jsonl').open('w',encoding='utf-8')
        writer=cv2.VideoWriter(str(out/'overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30.,(1280,720))
        if not writer.isOpened():raise RuntimeError('Could not create overlay.mp4')
        next_log=time.monotonic()
        if args.display:cv2.namedWindow('D455 RGB 2D',cv2.WINDOW_NORMAL)
        while not stop:
            try:frames=pipe.wait_for_frames(5000)
            except RuntimeError:
                if playback is not None and playback.current_status()==rs.playback_status.stopped:
                    status='complete';break
                raise
            frame=frames.get_color_frame()
            if not frame:continue
            number=int(frame.get_frame_number());t=float(frame.get_timestamp())*.001
            if last_number is not None and number<=last_number:duplicates+=1;continue
            if last_number is not None:frame_gaps+=max(0,number-last_number-1)
            last_number=number
            if first_t is None:first_t=t
            last_t=t
            bgr=np.asanyarray(frame.get_data()).copy();tic=time.perf_counter()
            result,candidates,_=detector.detect(bgr,t);timings.append((time.perf_counter()-tic)*1000)
            frame_count+=1
            rows.write(json.dumps(dict(frame_number=number,ball_2d=result))+'\n')
            if result['valid']:
                cv2.circle(bgr,tuple(np.rint(result['uv']).astype(int)),12,(0,255,0),2)
            cv2.putText(bgr,f"2D {result['state']} | RGB {frame_count} | gaps {frame_gaps}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
            cv2.putText(bgr,'green=current RGB measurement | no predicted points | depth=not requested',(10,49),cv2.FONT_HERSHEY_SIMPLEX,.47,(230,230,230),1)
            writer.write(bgr)
            if args.display:
                cv2.imshow('D455 RGB 2D',bgr)
                if cv2.waitKey(1)&0xff in (27,ord('q')):stop=True
            if time.monotonic()>=next_log:
                print(f'[RGB] {t-first_t:.1f}s frames={frame_count} gaps={frame_gaps} {result["state"]}',flush=True);next_log=time.monotonic()+2
            if args.duration and t-first_t>=args.duration:status='complete';break
        if stop:status='interrupted'
    except Exception as exc:
        error=str(exc);raise
    finally:
        if started:pipe.stop()
        if writer is not None:writer.release()
        if rows is not None:rows.close()
        if args.display:cv2.destroyAllWindows()
        signal.signal(signal.SIGINT,old_handler)
        duration=None if first_t is None or last_t is None else last_t-first_t
        summary=dict(status=status,error=error,serial=serial,frames=frame_count,
            sensor_duration_s=duration,frame_gaps=frame_gaps,duplicate_frames=duplicates,
            stream_fps=None if not duration else (frame_count-1)/duration,
            processing_ms=None if not timings else dict(median=float(np.median(timings)),p95=float(np.percentile(timings,95))),
            mode='RGB-only 2D; green points are image measurements, never depth or predictions',
            recorded_input=args.bag,
            algorithm_sha256=hashlib.sha256(Path(__file__).with_name('ball_image.py').read_bytes()).hexdigest(),
            note='Frame throughput is not detection recall. Recall requires labeled visible-ball frames.')
        (out/'run.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('[DONE] '+json.dumps(summary),flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--display',action='store_true');p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--output');p.add_argument('--serial');p.add_argument('--bag',help='Native RealSense recording for playback instead of opening the camera')
    args=p.parse_args()
    if args.duration<0:p.error('--duration must be nonnegative')
    try:run(args)
    except RuntimeError as exc:
        print(f'[ERROR] {exc}',file=sys.stderr)
        if 'busy' in str(exc).lower():print('Close the process currently using D455 before starting this program.',file=sys.stderr)
        sys.exit(1)
