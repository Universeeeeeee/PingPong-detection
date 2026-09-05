"""Per-stream acquisition gates. A successful pipeline.start is not readiness."""
import sqlite3
import threading

REQUIRED_STREAMS=('color_0','depth_0','infrared_1','infrared_2')


class StreamHealth:
    def __init__(self):
        self.counts={};self.last={};self.timestamps={};self.first=None
        self.lock=threading.Lock()

    def observe(self,name,host_time,sensor_time):
        with self.lock:
            self.counts[name]=self.counts.get(name,0)+1
            self.last[name]=host_time;self.timestamps[name]=sensor_time
            if self.first is None:self.first=host_time

    def snapshot(self,now):
        with self.lock:
            return dict(counts=dict(self.counts),ages_s={k:now-v for k,v in self.last.items()},
                        sensor_timestamps_s=dict(self.timestamps),first_host_s=self.first)

    def missing(self,now,max_age=.5,min_frames=5):
        s=self.snapshot(now)
        return [k for k in REQUIRED_STREAMS if s['counts'].get(k,0)<min_frames or s['ages_s'].get(k,float('inf'))>max_age]

    def ready(self,now):return not self.missing(now)


def raw_stream_inventory(path):
    """Read closed SDK ROS2 DB3 image counts without synchronized playback.

    This reveals a missing IR stream even when the SDK synchronizer emitted no
    framesets at all. Never infer completeness from raw-file size alone.
    """
    result=dict(counts={k:0 for k in REQUIRED_STREAMS},error=None)
    if not path.exists():result['error']='raw_file_missing';return result
    names={'Color_0':'color_0','Depth_0':'depth_0','Infrared_1':'infrared_1','Infrared_2':'infrared_2'}
    try:
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=3.) as db:
            for name,count in db.execute("SELECT t.name,COUNT(m.id) FROM topics t LEFT JOIN messages m ON m.topic_id=t.id WHERE t.name LIKE '%/image/data' GROUP BY t.id"):
                stream=name.split('/')[-3]
                if stream in names:result['counts'][names[stream]]=count
    except (sqlite3.Error,ValueError) as exc:result['error']=str(exc)
    return result
