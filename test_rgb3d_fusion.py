"""Integration regressions: image evidence may guide depth, never manufacture it."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import cv2
import numpy as np
import d455_table_tennis_tracker as app
from replay_ball_validation import make_system
from test_ball_validation import model,ball_image
import test_guided_depth


def system():
    s=make_system(model(),rgb_detector='image');s.table_tracker=None
    return s


class FusionTests(unittest.TestCase):
    def test_missing_depth_keeps_2d_without_creating_3d(self):
        s=system();b,_=ball_image()
        s._process_rgb_frame(b,None,1.)
        self.assertTrue(s.rgb_state.get_image_observation()['valid'])
        self.assertIsNone(s.rgb_state.get_measurement())
        packet=SimpleNamespace(timestamp_s=1.01,host_timestamp_s=1.,frame_number=1)
        payload=s._publish(packet,None,False,'prediction_only',None)
        self.assertTrue(payload['ball_2d']['valid']);self.assertFalse(payload['ball']['valid'])
        self.assertIsNone(payload['ball']['position_camera_m']);json.dumps(payload)

    def test_real_foreground_depth_is_available_for_fusion(self):
        s=system();b,d=ball_image();s._process_rgb_frame(b,d,1.,1.)
        m=s.rgb_state.get_measurement()
        self.assertIsNotNone(m);self.assertEqual(m.depth_source,'aligned_depth')
        self.assertAlmostEqual(m.position_left_camera[2],2.02,places=5)

    def test_bad_or_stale_depth_cannot_create_measurement(self):
        for radius,time in ((22,1.),(6,.9)):
            s=system();b,d=ball_image(radius=radius);s._process_rgb_frame(b,d,1.,time)
            self.assertTrue(s.rgb_state.get_image_observation()['valid'])
            self.assertIsNone(s.rgb_state.get_measurement())

    def test_rgb_is_published_once_and_expires(self):
        s=system();b,d=ball_image();s._process_rgb_frame(b,d,1.,1.)
        for t,new,valid in ((1.01,True,True),(1.02,False,True),(1.11,False,False)):
            packet=SimpleNamespace(timestamp_s=t,host_timestamp_s=t,frame_number=int(t*100))
            p=s._publish(packet,None,False,'prediction_only',None)['ball_2d']
            self.assertEqual(p['new_observation'],new);self.assertEqual(p['valid'],valid)
            self.assertEqual(p['timestamp_s'],1.)
        before=s.rgb_state.get_image_observation()
        s._process_rgb_frame(np.zeros_like(b),None,1.)
        self.assertEqual(s.rgb_state.get_image_observation(),before)
        self.assertEqual(s.rgb_processed_frames,1)

    def test_integrated_image_guides_actual_stereo_without_depth(self):
        s=system();b,_=ball_image()
        l=np.full((240,320),20,np.uint8);r=l.copy()
        cv2.circle(l,(160,120),6,100,-1);cv2.circle(r,(132,120),6,100,-1)
        for i,t in enumerate((1.,1.033,1.066,1.099)):
            s._process_rgb_frame(b,None,t)
            packet=SimpleNamespace(timestamp_s=t,host_timestamp_s=t,frame_number=i,ir_left=l,ir_right=r)
            measured,source,_,_=s._process_ball_packet(packet,None)
        self.assertTrue(measured);self.assertEqual(source,'ir_stereo')
        self.assertTrue(s.ball_gate.valid(t))
        self.assertAlmostEqual(s.camera_ball_tracker.filter.x[2],600*.095/28,places=4)

    def test_weak_image_cannot_initialize_stereo_identity(self):
        m,c,l,r=test_guided_depth.GuidedDepthTests().scene()
        c.validation['can_initialize_3d']=False
        lc=app.rgb_guided_ir_candidates(l,[c],m);rc=app.rgb_guided_ir_candidates(r,[c],m,True)
        found=app.StereoCandidateSelector(m,.02).select(lc,rc,timestamp_s=1.,
            rgb_observation=(1.,[c]),left_image=l,right_image=r)
        self.assertIsNone(found)

    def test_no_ir_contrast_cannot_initialize_from_2d_alone(self):
        s=system();b,_=ball_image();dark=np.full((240,320),16,np.uint8)
        for i in range(8):
            t=1+i/30;s._process_rgb_frame(b,None,t)
            packet=SimpleNamespace(timestamp_s=t,frame_number=i,ir_left=dark,ir_right=dark)
            measured,_,_,_=s._process_ball_packet(packet,None)
            self.assertFalse(measured)
        self.assertFalse(s.camera_ball_tracker.initialized)

    def test_stale_rgb_cannot_let_a_dark_shadow_follow_the_track(self):
        m,c,l,r=test_guided_depth.GuidedDepthTests().scene()
        lc=app.rgb_guided_ir_candidates(l,[c],m);rc=app.rgb_guided_ir_candidates(r,[c],m,True)
        selector=app.StereoCandidateSelector(m,.02)
        predicted=np.array([0.,0.,600*.095/28])
        for shadow in (False,True):
            a,b=(120-l,120-r) if shadow else (l,r)
            found=selector.select(lc,rc,predicted_position_left=predicted,timestamp_s=1.06,
                rgb_observation=(1.,[c]),tracking=True,left_image=a,right_image=b)
            if shadow:
                self.assertIsNone(found)
                self.assertIn('stale_rgb_requires_positive_ir_object',selector.diagnostics['rejected'])
            else:self.assertIsNotNone(found)


if __name__=='__main__':unittest.main(verbosity=2)
