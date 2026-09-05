import time
import unittest
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
from realtime_source import RealtimeSource,LatestQueue
from ball_image_roi import FastImageBallDetector
from ball_epipolar import EpipolarBallMatcher
from test_ball_validation import model
from fast_rgbd import FastRGBDProcessor


def image(x=160):
    b=np.full((240,480,3),(100,30,10),np.uint8)
    cv2.circle(b,(x,120),6,(0,140,255),-1);return b


class RealtimeTests(unittest.TestCase):
    def test_learned_hint_still_requires_an_observed_right_ball(self):
        from ball_foundation import FoundationBallMatcher
        class FakeRuntime:
            def warmup(self,*args):return dict(p95=0.)
            def predict(self,left,right):return np.full(left.shape,28.,np.float32)
        detector=FastImageBallDetector();r,_,_=detector.detect(image()[:,:320],1.)
        a=np.full((240,320),20,np.uint8);b=a.copy()
        cv2.circle(a,(160,120),6,100,-1);cv2.circle(b,(132,120),6,100,-1)
        matcher=FoundationBallMatcher(model(),runtime=FakeRuntime())
        self.assertIsNotNone(matcher.match(a,b,r,detector.selected_candidate,1.,1.))
        self.assertIsNone(matcher.match(a,np.full_like(b,20),r,detector.selected_candidate,1.01,1.01))

    def test_invalid_learned_depth_cannot_silently_use_unrestricted_ncc(self):
        from ball_foundation import FoundationBallMatcher
        class FakeRuntime:
            def warmup(self,*args):return dict(p95=0.)
            def predict(self,left,right):return np.full(left.shape,np.nan,np.float32)
        detector=FastImageBallDetector();r,_,_=detector.detect(image()[:,:320],1.)
        a=np.full((240,320),20,np.uint8);b=a.copy()
        cv2.circle(a,(160,120),6,100,-1);cv2.circle(b,(132,120),6,100,-1)
        matcher=FoundationBallMatcher(model(),runtime=FakeRuntime())
        self.assertIsNone(matcher.match(a,b,r,detector.selected_candidate,1.,1.))

    def test_learned_model_is_not_launched_after_its_budget(self):
        from ball_foundation import FoundationBallMatcher
        class FakeRuntime:
            def warmup(self,*args):return dict(p95=20.)
            def predict(self,*args):raise AssertionError('Late inference launched')
        detector=FastImageBallDetector();r,_,_=detector.detect(image()[:,:320],1.)
        a=np.full((240,320),20,np.uint8)
        matcher=FoundationBallMatcher(model(),runtime=FakeRuntime())
        self.assertIsNone(matcher.match(a,a,r,detector.selected_candidate,1.,1.,deadline=time.perf_counter()+.001))
        self.assertEqual(matcher.diagnostics['reason'],'ffs_budget_insufficient')

    def test_unrectified_calibration_is_rejected_before_inference(self):
        from ball_foundation import FoundationBallMatcher
        m=model();m.T_left_right[1,3]=.01
        with self.assertRaisesRegex(ValueError,'rectified'):
            FoundationBallMatcher(m,runtime=object())

    def test_ir_xyz_and_timestamp_are_not_replaced_by_rgb_ray(self):
        processor=FastRGBDProcessor(model());rgb=image(162)[:,:320]
        a=np.full((240,320),20,np.uint8);b=a.copy()
        cv2.circle(a,(160,120),6,100,-1);cv2.circle(b,(132,120),6,100,-1)
        pair=dict(t=.99,right_t=.9903,name='ir0',ir_left=a,ir_right=b)
        row=processor.process(rgb,1.,[pair])
        self.assertTrue(row['measured'])
        np.testing.assert_allclose(row['xyz'],row['stereo_measurement']['xyz_ir'])
        self.assertAlmostEqual(row['xyz'][0],0.,delta=.001)
        self.assertEqual(row['measurement_t'],.99)
        self.assertEqual(row['rgb_timestamp_s'],1.)
        self.assertEqual(row['stereo_right_t'],.9903)

    def test_unused_but_older_ir_cannot_move_state_backwards(self):
        processor=FastRGBDProcessor(model());rgb=image()[:,:320]
        a=np.full((240,320),20,np.uint8);b=a.copy()
        cv2.circle(a,(160,120),6,100,-1);cv2.circle(b,(132,120),6,100,-1)
        pair=dict(t=1.01,right_t=1.01,name='ir1',ir_left=a,ir_right=b)
        self.assertTrue(processor.process(rgb,1.,[pair])['measured'])
        row=processor.process(rgb,1.015,[dict(pair,t=1.005,right_t=1.005,name='ir0')])
        self.assertFalse(row['measured']);self.assertEqual(processor.last_depth[0],1.01)

    def test_coverage_uses_rgb_identity_timestamp_not_ir_timestamp(self):
        from replay_realtime_3d import score_outputs
        with tempfile.TemporaryDirectory(prefix='d455_score_') as tmp:
            root=Path(tmp);(root/'analysis/rgb_2d_audit').mkdir(parents=True)
            (root/'camera.json').write_text(json.dumps(dict(action_start_sensor_timestamp_s=1.,requested_duration_s=1.)))
            (root/'analysis/rgb_2d_audit/labels.json').write_text(json.dumps([
                dict(frame='rgb0',status='visible',uv=[160,120],tolerance_px=10)]))
            events=[dict(kind='rgb',name='rgb0',t=1.),dict(kind='rgb',name='rgb1',t=1.033)]
            rows=[dict(measured=True,xyz=[0,0,2],rgb_uv=[160,120],measurement_t=1.017,
                       rgb_timestamp_s=1.,latency_ms=30)]
            score=score_outputs(root,events,rows,50)
            self.assertEqual(score['within_deadline_3d_frames'],1)

    def test_same_stereo_pair_cannot_count_as_two_new_depth_measurements(self):
        processor=FastRGBDProcessor(model());rgb=image()[:,:320]
        a=np.full((240,320),20,np.uint8);b=a.copy()
        cv2.circle(a,(160,120),6,100,-1);cv2.circle(b,(132,120),6,100,-1)
        pair=dict(t=1.,right_t=1.,name='ir0',ir_left=a,ir_right=b)
        first=processor.process(rgb,1.,[pair]);self.assertTrue(first['measured'])
        second=processor.process(rgb,1.01,[pair]);self.assertFalse(second['measured'])
        third=processor.process(rgb,1.02,[dict(pair,t=1.02,right_t=1.02,name='ir1')])
        self.assertTrue(third['measured'])

    def test_expired_budget_cannot_create_or_reuse_depth(self):
        processor=FastRGBDProcessor(model())
        row=processor.process(image()[:,:320],1.,[],deadline=time.perf_counter()-1)
        self.assertFalse(row['measured']);self.assertIsNone(row['xyz'])
        self.assertIsNone(processor.image.engine.last_timestamp)

    def test_queue_drops_old_frame(self):
        q=LatestQueue();q.put(1);q.put(2);self.assertEqual(q.get(),2);self.assertEqual(q.dropped,1)

    def test_source_never_delivers_before_availability(self):
        events=[dict(t=1.,available_t=1.01,kind='rgb'),dict(t=1.005,available_t=1.02,kind='ir')]
        source=RealtimeSource(events);seen=[]
        source.run(lambda e:seen.append(time.perf_counter()-source.capture_wall(e['available_t'])))
        self.assertEqual(len(seen),2);self.assertTrue(all(t>=0 for t in seen))

    def test_roi_preserves_global_pixel_coordinates(self):
        d=FastImageBallDetector()
        for i,x in enumerate((160,170,180,190)):
            r,_,_=d.detect(image(x),1+i/30)
            self.assertTrue(r['valid']);np.testing.assert_allclose(r['uv'],[x,120],atol=.5)
        self.assertEqual(d.mode,'roi')
        self.assertEqual(d.engine.background.shape,(240,480,3))

    def test_displaced_target_causes_full_frame_recovery(self):
        d=FastImageBallDetector();d.detect(image(100),1.)
        r,_,_=d.detect(image(400),1.033)
        self.assertTrue(r['valid']);self.assertEqual(d.mode,'fallback')
        np.testing.assert_allclose(r['uv'],[400,120],atol=.5)

    def test_duplicate_cannot_mutate_roi_history(self):
        d=FastImageBallDetector();d.detect(image(),1.);bg=d.engine.background.copy()
        r,_,_=d.detect(image(300),1.)
        self.assertFalse(r['valid']);np.testing.assert_array_equal(bg,d.engine.background)

    def test_patch_matcher_requires_two_observed_objects(self):
        m=model();d=FastImageBallDetector();r,_,_=d.detect(image()[:,:320],1.)
        a=np.full((240,320),20,np.uint8);b=a.copy()
        cv2.circle(a,(160,120),6,100,-1);cv2.circle(b,(132,120),6,100,-1)
        matcher=EpipolarBallMatcher(m)
        match=matcher.match(a,b,r,d.selected_candidate,1.,1.)
        self.assertIsNotNone(match);self.assertAlmostEqual(match['xyz_ir'][2],600*.095/28,delta=.06)
        self.assertIsNone(matcher.match(a,np.full_like(b,20),r,d.selected_candidate,1.,1.))
        self.assertIsNone(matcher.match(a,b,r,d.selected_candidate,1.05,1.))

    def test_weak_but_resolved_ball_is_distinct_from_independent_noise(self):
        m=model();d=FastImageBallDetector();r,_,_=d.detect(image()[:,:320],1.)
        a=np.full((240,320),16,np.uint8);b=a.copy()
        cv2.circle(a,(160,120),6,18,-1);cv2.circle(b,(132,120),6,18,-1)
        matcher=EpipolarBallMatcher(m)
        self.assertIsNotNone(matcher.match(a,b,r,d.selected_candidate,1.,1.))
        for seed in range(4):
            rng=np.random.default_rng(seed)
            l=np.uint8(16+np.minimum(2,rng.poisson(.15,a.shape)))
            rr=np.uint8(16+np.minimum(2,rng.poisson(.15,a.shape)))
            self.assertIsNone(matcher.match(l,rr,r,d.selected_candidate,1.,1.))

    def test_large_ir_object_cannot_pass_as_ball_sized_texture(self):
        m=model();d=FastImageBallDetector();r,_,_=d.detect(image()[:,:320],1.)
        a=np.full((240,320),16,np.uint8);b=a.copy()
        cv2.circle(a,(160,120),24,150,-1);cv2.circle(b,(132,120),24,150,-1)
        self.assertIsNone(EpipolarBallMatcher(m).match(a,b,r,d.selected_candidate,1.,1.))


if __name__=='__main__':unittest.main(verbosity=2)
