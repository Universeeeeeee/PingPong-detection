"""Causal, wall-clock replay of decoded D455 arrays at recorded arrival times."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path
import queue
import threading
import time
import numpy as np


class LatestQueue:
    def __init__(self):self.q=queue.Queue(1);self.dropped=0
    def put(self,item):
        try:self.q.put_nowait(item)
        except queue.Full:
            try:self.q.get_nowait();self.dropped+=1
            except queue.Empty:pass
            self.q.put_nowait(item)
    def get(self,timeout=.05):return self.q.get(timeout=timeout)


def load_recording(root):
    root=Path(root);arrival={};raw_metadata=root/'frames.jsonl'
    if raw_metadata.exists():
        for line in raw_metadata.open():
            r=json.loads(line);toa=r.get('sdk_metadata',{}).get('time_of_arrival')
            if toa is not None:arrival[(r['stream'],r['frame_number'])]=float(toa)*.001
    def read(path):
        with np.load(path) as f:
            kind='rgb' if path.name.startswith('rgb') else 'ir'
            keys=('bgr','depth') if kind=='rgb' else ('ir_left','ir_right')
            row={k:f[k].copy() for k in keys};row.update(kind=kind,name=path.name,
                t=float(f['timestamp_s']),frame_number=int(f['frame_number']))
            if kind=='rgb':row['depth_t']=float(f['depth_timestamp_s'])
            else:row['right_t']=float(f['right_timestamp_s'])
        streams=['color_0'] if kind=='rgb' else ['infrared_1','infrared_2']
        a=[arrival.get((s,row['frame_number']),row['t']) for s in streams]
        row['available_t']=max(row['t'],*a)
        return row
    paths=sorted((root/'frames').glob('rgb_*.npz'))+sorted((root/'frames').glob('ir_*.npz'))
    with ThreadPoolExecutor(max_workers=4) as pool:events=list(pool.map(read,paths))
    events.sort(key=lambda r:r['available_t'])
    return events


class RealtimeSource:
    def __init__(self,events):
        self.events=events;self.first=min(e['t'] for e in events)
        self.start_wall=None;self.done=threading.Event();self.cancel=threading.Event()
        self.jitter_ms=[];self.delivered=Counter()
    def capture_wall(self,t):return self.start_wall+(t-self.first)
    def run(self,consume):
        self.start_wall=time.perf_counter()+.1
        try:
            for e in self.events:
                deadline=self.capture_wall(e['available_t'])
                if self.cancel.wait(max(0,deadline-time.perf_counter())):break
                self.jitter_ms.append(max(0,1000*(time.perf_counter()-deadline)))
                consume(e);self.delivered[e['kind']]+=1
        finally:self.done.set()


def percentiles(values):
    if not values:return None
    return dict(zip(('p50','p95','p99','max'),map(float,np.percentile(values,[50,95,99,100]))))
