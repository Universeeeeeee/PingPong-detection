"""Bounded native RealSense recording for offline motion evaluation (no detector).

Raw RGB, both IR and depth plus calibration/timestamps are stored by the SDK in
capture.db3. Preview drops never substitute for raw-frame recording counts.
"""
import argparse
import datetime
import fcntl
import json
import math
from pathlib import Path
import queue
import shutil
import signal
import sys
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from d455_table_tennis_tracker import D455AsyncCapture
from recording_health import StreamHealth, REQUIRED_STREAMS, raw_stream_inventory


class RecordingCapture(D455AsyncCapture):
    def __init__(self, path, args):
        super().__init__(args.serial,args.ir_fps,1280,720,30,4,True,args.ir_exposure_us,0.,ir_gain=args.ir_gain)
        self.path=path
        self.metadata_queue=queue.Queue(maxsize=8192)
        self.preview_queue=queue.Queue(maxsize=2)
        self.last_numbers={};self.counts={};self.gaps={}
        self.metadata_drops=0;self.preview_drops=0
        self.health=StreamHealth();self.last_callback_error=None

    def _make_config(self, fps, size):
        cfg=super()._make_config(fps,size)
        cfg.enable_record_to_file(str(self.path))
        return cfg

    def _candidate_ir_fps(self, requested):
        # A requested exposure/frame-rate comparison must not silently change fps.
        return [requested]

    def _frame_callback(self, frame):
        try:
            for f in frame.as_frameset():
                p=f.get_profile();name=str(p.stream_type()).split('.')[-1]+'_'+str(p.stream_index())
                number=int(f.get_frame_number())
                previous=self.last_numbers.get(name)
                if previous is not None and number<=previous:continue
                self.last_numbers[name]=number
                self.counts[name]=self.counts.get(name,0)+1
                self.health.observe(name,time.monotonic(),f.get_timestamp()*.001)
                self.gaps[name]=self.gaps.get(name,0)+(max(0,number-previous-1) if previous is not None else 0)
                row=dict(stream=name,frame_number=number,timestamp_s=f.get_timestamp()*.001,
                         timestamp_domain=str(f.get_frame_timestamp_domain()),host_monotonic_s=time.monotonic())
                values={}
                for key in ('actual_exposure','gain_level','frame_timestamp','sensor_timestamp',
                            'backend_timestamp','time_of_arrival','actual_fps','auto_exposure',
                            'sequence_id','sequence_size'):
                    option=getattr(rs.frame_metadata_value,key,None)
                    if option is not None and f.supports_frame_metadata(option):
                        values[key]=f.get_frame_metadata(option)
                row['sdk_metadata']=values
                try:self.metadata_queue.put_nowait(row)
                except queue.Full:self.metadata_drops+=1
                if p.stream_type()==rs.stream.color:
                    image=np.asanyarray(f.get_data()).copy()
                    try:self.preview_queue.put_nowait((row['timestamp_s'],image))
                    except queue.Full:self.preview_drops+=1
        except Exception as exc:
            self.callback_errors+=1
            self.last_callback_error=repr(exc)


def camera_metadata(capture, model):
    def intr(i):
        return {**{k:getattr(i,k) for k in ('width','height','fx','fy','ppx','ppy')},
                'model':str(i.model),'coeffs':list(i.coeffs)}
    data={k:intr(getattr(model,k)) for k in ('ir_left_intr','ir_right_intr','color_intr')}
    data.update({k:getattr(model,k).tolist() for k in ('T_left_right','T_right_left','T_left_color','T_color_left')})
    data.update(depth_scale=model.depth_scale,selected_ir_fps=model.selected_ir_fps,
                selected_color_size=model.selected_color_size)
    device=capture.profile.get_device()
    data['serial']=device.get_info(rs.camera_info.serial_number)
    data['device']=device.get_info(rs.camera_info.name)
    data['sensors']=[]
    for sensor in device.query_sensors():
        options={}
        for key in ('exposure','gain','enable_auto_exposure','emitter_enabled','emitter_on_off','hdr_enabled'):
            option=getattr(rs.option,key,None)
            if option is not None and sensor.supports(option):options[key]=sensor.get_option(option)
        data['sensors'].append(dict(name=sensor.get_info(rs.camera_info.name),options=options))
    return data


def reset_recording_device(serial,stop,timeout=15.):
    """One bounded recovery, after our capture has stopped and released USB."""
    ctx=rs.context();devices=[d for d in ctx.query_devices() if serial is None or d.get_info(rs.camera_info.serial_number)==serial]
    if len(devices)!=1:raise RuntimeError('Cannot safely select the D455 for recovery')
    serial=devices[0].get_info(rs.camera_info.serial_number)
    print('[RECOVERY] Reinitializing D455 '+serial,flush=True)
    devices[0].hardware_reset();devices.clear()
    start=time.monotonic()
    while not stop.wait(.25):
        elapsed=time.monotonic()-start
        if elapsed>=timeout:raise RuntimeError('D455 did not reconnect within %.0f seconds'%timeout)
        if elapsed>=2. and any(d.get_info(rs.camera_info.serial_number)==serial for d in ctx.query_devices()):return serial
    raise InterruptedError('Recording cancelled during recovery')


def record(args):
    base=Path(__file__).resolve().parent
    lock=(base/'.d455_tracker.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('D455 tracker/recorder already running. Stop it with Ctrl+C first.')
    parent=Path(args.output_root).expanduser().resolve();parent.mkdir(parents=True,exist_ok=True)
    directory=parent/(datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+args.scene)
    directory.mkdir(exist_ok=False)
    reserve=2_000_000_000
    if shutil.disk_usage(directory).free < reserve+int((args.duration+args.warmup)*250_000_000):
        raise RuntimeError('Insufficient free space for the requested raw recording duration')
    capture=RecordingCapture(directory/'capture.db3',args)
    stop=threading.Event();capture.cancel_event=stop
    previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    for sig in previous_handlers:signal.signal(sig,lambda *_:stop.set())
    cv2.setNumThreads(1)
    writer=None;video_start=None;video_frames=0;model=None
    meta=dict(scene=args.scene,requested_duration_s=args.duration,warmup_s=args.warmup,
              started_at=datetime.datetime.now().astimezone().isoformat(),status='starting',
              preview_fps=15,raw_file='capture.db3',recording_includes_warmup=True,
              recording_includes_startup=True,preview_omits_startup=True,startup_attempts=[],
              requested_ir_exposure_us=args.ir_exposure_us,requested_ir_gain=args.ir_gain)
    metadata_file=(directory/'frames.jsonl').open('w')
    print('[INFO] Recording directory:',directory,flush=True)
    def drain_metadata():
        while True:
            try:row=capture.metadata_queue.get_nowait()
            except queue.Empty:break
            metadata_file.write(json.dumps(row)+'\n')
    def save_metadata():
        temp=directory/'camera.json.tmp';temp.write_text(json.dumps(meta,indent=2)+'\n');temp.replace(directory/'camera.json')
    window='D455 raw RGB recording - Q to stop'
    last_display_image=None
    def poll_display():
        if cv2.waitKey(1)&255 in (ord('q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:stop.set()
    def display(image,text):
        nonlocal last_display_image
        if not args.display:return
        if image is not None:last_display_image=image
        canvas=np.zeros((720,1280,3),np.uint8) if last_display_image is None else last_display_image.copy()
        cv2.rectangle(canvas,(0,0),(canvas.shape[1],46),(25,25,25),-1)
        cv2.putText(canvas,text,(12,31),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,230,255),2)
        cv2.imshow(window,canvas)
        poll_display()
    def check_callback():
        if capture.callback_errors:raise RuntimeError('Capture callback failed: '+str(capture.last_callback_error))
        if capture.metadata_drops:raise RuntimeError('Metadata queue overflow; raw acquisition needs verification')
    started=time.monotonic();last_status=started;failure=None;ready=False
    save_metadata()
    try:
        display(None,'WAITING: checking RGB / depth / left IR / right IR')
        for attempt in range(args.startup_retries+1):
            if stop.is_set():raise InterruptedError('Recording cancelled before startup')
            if attempt:
                capture=RecordingCapture(directory/'capture.db3',args);capture.cancel_event=stop
            started=time.monotonic();model=capture.start();meta.update(camera_metadata(capture,model))
            meta['capture_start_host_monotonic_s']=started
            wait_started=time.monotonic();last_wait=0.
            while not capture.health.ready(time.monotonic()):
                if stop.is_set():raise InterruptedError('Recording cancelled while waiting for streams')
                now=time.monotonic();check_callback();drain_metadata()
                missing=capture.health.missing(now)
                try:_,image=capture.preview_queue.get_nowait()
                except queue.Empty:image=None
                display(image,'WAITING: '+', '.join(missing))
                if now-last_wait>=1.:
                    last_wait=now;print('[WAIT] Missing/unsettled streams: '+', '.join(missing),flush=True)
                if now-wait_started>=args.startup_timeout:break
                stop.wait(.025)
            if capture.health.ready(time.monotonic()):
                meta['startup_attempts'].append(dict(attempt=attempt+1,status='ready',counts=dict(capture.counts)))
                break
            capture.stop();drain_metadata();inventory=raw_stream_inventory(directory/'capture.db3')
            absent=[k for k in REQUIRED_STREAMS if not inventory['counts'].get(k,0)]
            why='Startup timed out; raw missing streams: '+', '.join(absent or ['synchronized frames'])
            info=dict(attempt=attempt+1,status='failed',error=why,raw=inventory,callback_counts=dict(capture.counts))
            meta['startup_attempts'].append(info);save_metadata()
            print('[ERROR] '+why,flush=True)
            if attempt>=args.startup_retries:raise RuntimeError(why+'; recovery limit reached')
            # Preserve failed acquisitions separately; never overwrite their raw data.
            metadata_file.close();archive=directory/('failed_attempt_%02d'%(attempt+1));archive.mkdir()
            for name in ('capture.db3','frames.jsonl'):
                path=directory/name
                if path.exists():path.replace(archive/name)
            (archive/'diagnostics.json').write_text(json.dumps(info,indent=2)+'\n')
            metadata_file=(directory/'frames.jsonl').open('w')
            args.serial=reset_recording_device(meta.get('serial',args.serial),stop)
        if stop.is_set():raise InterruptedError('Recording cancelled before warmup')
        verified=time.monotonic();meta['streams_verified_host_monotonic_s']=verified
        started=verified;last_status=started
        meta['action_start_host_monotonic_s']=started+args.warmup
        meta['status']='warming_up';save_metadata()
        print('[OK] All four streams are arriving:',dict(capture.counts),flush=True)
        print('[INFO] Keep camera fixed. Warmup/countdown:',args.warmup,'seconds.',flush=True)
        while not stop.is_set():
            now=time.monotonic();elapsed=now-started
            check_callback()
            stale=capture.health.missing(now,max_age=args.frame_timeout,min_frames=1)
            if stale:raise RuntimeError('Stream stalled for %.1f seconds: %s'%(args.frame_timeout,', '.join(stale)))
            if not ready and elapsed>=args.warmup:
                ready=True;meta['status']='recording';meta['action_counts_at_start']=dict(capture.counts)
                meta['action_start_host_monotonic_s']=now
                health=capture.health.snapshot(now)
                meta['action_start_sensor_timestamp_s']=health['sensor_timestamps_s']['infrared_1']+health['ages_s']['infrared_1']
                save_metadata();print('[READY] Start '+args.scene+' now.',flush=True)
            if ready and now-meta['action_start_host_monotonic_s']>=args.duration:break
            drain_metadata()
            try:t,bgr=capture.preview_queue.get(timeout=.025)
            except queue.Empty:t,bgr=None,None
            if bgr is not None:
                if writer is None:
                    writer=cv2.VideoWriter(str(directory/'preview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15.,(bgr.shape[1],bgr.shape[0]))
                    if not writer.isOpened():raise RuntimeError('Cannot open preview writer')
                    video_start=t
                target=int(max(0.,t-video_start)*15)+1
                while video_frames<target:writer.write(bgr);video_frames+=1
                if args.display:
                    display(bgr,('RECORDING %.1f / %.1f s'%(max(0.,elapsed-args.warmup),args.duration)) if ready else ('WARMUP %.1f s'%max(0.,args.warmup-elapsed)))
            elif args.display:
                # A 25 ms queue wait can expire between healthy 30 Hz RGB frames.
                # Keep the last image instead of flashing a black placeholder.
                poll_display()
            if now-last_status>=2.:
                last_status=now
                print('[%s] %.1f/%.1f s; stream frames=%s; metadata drops=%d' %
                      ('REC' if ready else 'WARMUP',max(0.,elapsed-args.warmup),args.duration,capture.counts,capture.metadata_drops),flush=True)
                metadata_file.flush()
                meta['callback_stream_counts']=dict(capture.counts);save_metadata()
                if shutil.disk_usage(directory).free<reserve:raise RuntimeError('Disk reserve reached; recording stopped')
        meta['status']='interrupted' if stop.is_set() else 'finalizing'
    except InterruptedError:
        meta['status']='interrupted'
    except Exception as exc:
        failure=exc;meta.update(status='failed',error=str(exc))
    finally:
        capture_duration=time.monotonic()-meta.get('capture_start_host_monotonic_s',started)
        stop.set();capture.stop();drain_metadata();metadata_file.close()
        if writer is not None:writer.release()
        if args.display:cv2.destroyAllWindows()
        meta.update(actual_capture_duration_s=capture_duration,
                    callback_stream_counts=capture.counts,callback_frame_gaps=capture.gaps,
                    metadata_drops=capture.metadata_drops,preview_drops=capture.preview_drops,
                    callback_errors=capture.callback_errors,preview_frames=video_frames,
                    last_callback_error=capture.last_callback_error,
                    raw_bytes=(directory/'capture.db3').stat().st_size if (directory/'capture.db3').exists() else 0,
                    validation_note='Callback counts are acquisition diagnostics. Use SDK playback to verify raw recording completeness.')
        meta['raw_stream_inventory']=raw_stream_inventory(directory/'capture.db3')
        if meta['status']=='finalizing':
            raw=meta['raw_stream_inventory']
            missing=[k for k in REQUIRED_STREAMS if not raw['counts'].get(k,0) or not capture.counts.get(k,0)]
            action_counts={k:capture.counts.get(k,0)-meta.get('action_counts_at_start',{}).get(k,0) for k in REQUIRED_STREAMS}
            meta['action_stream_counts']=action_counts
            slow=[k for k,n in action_counts.items() if n<args.duration*(30 if k=='color_0' else args.ir_fps)*.85]
            if missing or slow or raw['error'] or capture.callback_errors or capture.metadata_drops:
                failure=RuntimeError('Recording validation failed: missing=%s, low_rate=%s, raw_error=%s'%(missing,slow,raw['error']))
                meta.update(status='failed',error=str(failure))
            else:meta['status']='complete'
        save_metadata()
        lock.close()
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
        print('[DONE]',directory,'status='+meta['status'],flush=True)
    if failure:raise failure
    return directory


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--duration',type=float,default=30.,help='Action duration, excluding warmup (seconds)')
    p.add_argument('--warmup',type=float,default=3.)
    p.add_argument('--scene',choices=['rally','bounce','fast','occlusion','no_ball','static','smoke'],default='rally')
    p.add_argument('--output-root',default=str(Path(__file__).resolve().parent/'recordings'))
    p.add_argument('--serial');p.add_argument('--ir-fps',type=int,choices=[30,60,90],default=90)
    p.add_argument('--ir-exposure-us',type=float,default=800.)
    p.add_argument('--ir-gain',type=float,default=None,help='Manual IR gain; omitted preserves sensor setting')
    p.add_argument('--display',action='store_true')
    p.add_argument('--startup-timeout',type=float,default=6.,help='Seconds to wait for all four streams before recovery')
    p.add_argument('--frame-timeout',type=float,default=2.,help='Stop recording if any stream stalls this long')
    p.add_argument('--startup-retries',type=int,choices=[0,1],default=1,help='At most one device reset after missing startup streams')
    return p


if __name__=='__main__':
    p=parser();args=p.parse_args()
    if not math.isfinite(args.duration) or not 0<args.duration<=600:p.error('duration must be in (0,600]')
    if not math.isfinite(args.warmup) or not 0<=args.warmup<=30:p.error('warmup must be in [0,30]')
    if not math.isfinite(args.ir_exposure_us) or args.ir_exposure_us<0:p.error('exposure must be finite and >=0')
    if args.ir_gain is not None and (not math.isfinite(args.ir_gain) or args.ir_gain<=0):p.error('gain must be finite and >0')
    if any(not math.isfinite(v) or v<=0 for v in (args.startup_timeout,args.frame_timeout)):p.error('stream timeouts must be finite and positive')
    try:record(args)
    except Exception as exc:
        print('[ERROR] '+str(exc),file=sys.stderr,flush=True)
        raise SystemExit(1)
