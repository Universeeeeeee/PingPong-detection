import contextlib
import io
import json
from pathlib import Path
import queue
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from recording_health import StreamHealth,REQUIRED_STREAMS,raw_stream_inventory
import record_d455_motion as recorder


class HealthTests(unittest.TestCase):
    def test_requires_every_stream_and_multiple_fresh_frames(self):
        h=StreamHealth()
        for k in REQUIRED_STREAMS[:-1]:
            for n in range(5):h.observe(k,1.,1.)
        self.assertFalse(h.ready(1.));self.assertIn('infrared_2',h.missing(1.))
        for n in range(4):h.observe('infrared_2',1.,1.)
        self.assertFalse(h.ready(1.));h.observe('infrared_2',1.,1.)
        self.assertTrue(h.ready(1.));self.assertFalse(h.ready(1.6))

    def test_one_stalled_stream_invalidates_running_acquisition(self):
        h=StreamHealth()
        for k in REQUIRED_STREAMS:h.observe(k,1.,1.)
        for k in REQUIRED_STREAMS[:-1]:h.observe(k,4.,4.)
        self.assertEqual(h.missing(4.,max_age=2.,min_frames=1),['infrared_2'])

    def test_raw_inventory_detects_missing_ir_despite_data_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'capture.db3'
            with sqlite3.connect(p) as db:
                db.executescript('CREATE TABLE topics(id INTEGER,name TEXT); CREATE TABLE messages(id INTEGER,topic_id INTEGER);')
                db.executemany('INSERT INTO topics VALUES (?,?)',[(1,'/device_0/sensor_0/Depth_0/image/data'),(2,'/device_0/sensor_1/Color_0/image/data')])
                db.executemany('INSERT INTO messages VALUES (?,?)',[(1,1),(2,1),(3,2)])
            r=raw_stream_inventory(p)
            self.assertEqual(r['counts'],dict(color_0=1,depth_0=2,infrared_1=0,infrared_2=0))


class FakeCapture:
    initially_ready=False
    def __init__(self,path,args):
        self.path=path;self.health=StreamHealth();self.counts={};self.gaps={}
        self.callback_errors=self.metadata_drops=self.preview_drops=0
        self.last_callback_error=None;self.metadata_queue=queue.Queue();self.preview_queue=queue.Queue()
    def start(self):
        self.path.write_bytes(b'preserved raw data')
        if self.initially_ready:
            for k in REQUIRED_STREAMS:
                for _ in range(5):self.health.observe(k,time.monotonic(),1.)
                self.counts[k]=5
        return SimpleNamespace()
    def stop(self):pass


class RecorderFlowTests(unittest.TestCase):
    def run_case(self,args,cls=FakeCapture,event=None):
        self.output=io.StringIO()
        with patch.object(recorder,'RecordingCapture',cls),patch.object(recorder,'camera_metadata',return_value={'serial':'test'}),patch.object(recorder,'reset_recording_device',return_value='test') as reset,contextlib.redirect_stdout(self.output):
            if event is None:
                try:recorder.record(args)
                except RuntimeError:pass
            else:
                with patch.object(recorder.threading,'Event',return_value=event):recorder.record(args)
            self.reset_count=reset.call_count
        directory=next(Path(args.output_root).iterdir())
        return directory,json.loads((directory/'camera.json').read_text())

    def args(self,tmp):
        return recorder.parser().parse_args(['--output-root',tmp,'--duration','.1','--warmup','0','--startup-timeout','.015','--frame-timeout','.01'])

    def test_empty_streams_never_ready_and_retry_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            d,m=self.run_case(self.args(tmp))
            self.assertEqual(m['status'],'failed');self.assertEqual(self.reset_count,1)
            self.assertEqual(len(m['startup_attempts']),2)
            self.assertNotIn('[READY]',self.output.getvalue())
            self.assertNotIn('status=complete',self.output.getvalue())
            self.assertEqual((d/'failed_attempt_01'/'capture.db3').read_bytes(),b'preserved raw data')

    def test_cancel_before_frames_is_clean_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            event=threading.Event();event.set();d,m=self.run_case(self.args(tmp),event=event)
            self.assertEqual(m['status'],'interrupted');self.assertEqual(self.reset_count,0)
            self.assertNotIn('[READY]',self.output.getvalue())

    def test_running_stream_stall_fails_without_reconnecting_mid_action(self):
        class StallingCapture(FakeCapture):initially_ready=True
        with tempfile.TemporaryDirectory() as tmp:
            d,m=self.run_case(self.args(tmp),StallingCapture)
            self.assertEqual(m['status'],'failed');self.assertEqual(self.reset_count,0)
            self.assertIn('Stream stalled',m['error'])

    def test_healthy_rgb_interval_does_not_flash_black_window(self):
        class StallingCapture(FakeCapture):initially_ready=True
        with tempfile.TemporaryDirectory() as tmp,patch.object(recorder.cv2,'imshow') as show,patch.object(recorder.cv2,'waitKey',return_value=-1),patch.object(recorder.cv2,'getWindowProperty',return_value=1),patch.object(recorder.cv2,'destroyAllWindows'):
            args=self.args(tmp);args.display=True
            self.run_case(args,StallingCapture)
            self.assertEqual(show.call_count,1)  # Startup image only; no black replacement on queue timeout.

    def test_callback_exception_is_reported_instead_of_swallowed(self):
        class BrokenCapture(FakeCapture):
            def start(self):
                result=super().start();self.callback_errors=1
                self.last_callback_error='example metadata failure';return result
        with tempfile.TemporaryDirectory() as tmp:
            _,m=self.run_case(self.args(tmp),BrokenCapture)
            self.assertEqual(m['status'],'failed');self.assertEqual(self.reset_count,0)
            self.assertIn('example metadata failure',m['error'])


if __name__=='__main__':unittest.main()
