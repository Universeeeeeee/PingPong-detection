import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np
import pyrealsense2 as rs

from ball_validation import BallTrackGate,BallValidationConfig,contour_features,depth_evidence,size_check
import d455_table_tennis_tracker as app


def model():
    i=rs.intrinsics();i.width=320;i.height=240;i.fx=i.fy=600.;i.ppx=160.;i.ppy=120.
    i.model=rs.distortion.none;i.coeffs=[0.]*5
    lr=np.eye(4);lr[0,3]=-.095
    return SimpleNamespace(color_intr=i,ir_left_intr=i,ir_right_intr=i,
        T_left_color=np.eye(4),T_color_left=np.eye(4),T_left_right=lr,T_right_left=np.linalg.inv(lr),
        depth_scale=.001,selected_ir_fps=90,selected_color_size=(320,240))


def system():
    from replay_ball_validation import make_system
    s=make_system(model());s.table_tracker=None
    return s


def ball_image(radius=6,background_depth=2080,ball_depth=2000):
    b=np.full((240,320,3),(100,30,10),np.uint8)
    d=np.full((240,320),background_depth,np.uint16)
    cv2.circle(b,(160,120),radius,(0,140,255),-1)
    cv2.circle(d,(160,120),radius,ball_depth,-1)
    return b,d


def candidate(x,y,width=12.,features=None):
    return app.BallCandidate2D(np.array([x,y],float),width/2,(int(x-width/2),int(y-width/2),int(width),int(width)),
                             .8,100.,1.,1.,'test',features=features or {'minor_px':width,'major_px':width})


class EvidenceTests(unittest.TestCase):
    def test_static_ball_accepted_without_motion(self):
        s=system();b,d=ball_image()
        cs=s._process_rgb_frame(b,d,1.,1.)
        self.assertTrue(cs[0].validation['identity_ok'])
        self.assertEqual(s.rgb_state.get_measurement().depth_source,'aligned_depth')

    def test_tiny_color_fragment_cannot_initialize(self):
        s=system();b,d=ball_image(radius=1)
        s._process_rgb_frame(b,d,1.,1.)
        self.assertIsNone(s.rgb_state.get_measurement())

    def test_flat_background_depth_never_validates_ball_size(self):
        s=system();b,d=ball_image(background_depth=2000)
        cs=s._process_rgb_frame(b,d,1.,1.)
        self.assertTrue(cs[0].validation['identity_ok'])
        self.assertEqual(cs[0].validation['depth_source'],'stereo_required')
        self.assertIsNone(s.rgb_state.get_measurement())

    def test_missing_depth_does_not_fabricate_3d(self):
        s=system();b,d=ball_image();d[:]=0
        s._process_rgb_frame(b,d,1.,1.)
        self.assertIsNone(s.rgb_state.get_measurement())

    def test_wrong_physical_size_rejected(self):
        s=system();b,d=ball_image(radius=22)
        cs=s._process_rgb_frame(b,d,1.,1.)
        self.assertEqual(cs[0].validation['rejection_reason'],'physical_size')
        self.assertIsNone(s.rgb_state.get_measurement())

    def test_orange_fragment_on_red_basket_is_not_independent_ball(self):
        s=system();b,d=ball_image()
        b[90:150,130:190]=(0,0,255);cv2.circle(b,(160,120),6,(0,140,255),-1)
        d[:]=2000
        cs=s._process_rgb_frame(b,d,1.,1.)
        self.assertEqual(cs[0].validation['rejection_reason'],'embedded_color_unresolved')

    def test_nearby_same_color_can_be_resolved_by_foreground_depth(self):
        s=system();b,d=ball_image()
        b[90:150,130:190]=(0,0,255);cv2.circle(b,(160,120),6,(0,140,255),-1)
        cs=s._process_rgb_frame(b,d,1.,1.)
        self.assertTrue(cs[0].validation['identity_ok'])

    def test_mixed_and_stale_depth_cannot_initialize(self):
        s=system();b,d=ball_image();d[115:125:2,155:165]=3500
        s._process_rgb_frame(b,d,1.,1.);self.assertIsNone(s.rgb_state.get_measurement())
        b,d=ball_image();s._process_rgb_frame(b,d,2.,1.9)
        self.assertIsNone(s.rgb_state.get_measurement())
        b,d=ball_image();d[:]=1000
        cs=s._process_rgb_frame(b,d,3.,2.9)
        self.assertTrue(cs[0].validation['identity_ok'])

    def test_stationary_elongated_object_rejected_moving_streak_retained(self):
        s=system();b,d=ball_image();b[:]=(100,30,10);d[:]=0
        s._process_rgb_frame(b,d,1.,1.)
        cv2.line(b,(150,120),(180,120),(0,140,255),8)
        cs=s._process_rgb_frame(b,d,1.03,1.03)
        self.assertTrue(cs[0].validation['identity_ok'])
        cs=s._process_rgb_frame(b,d,1.06,1.06)
        self.assertEqual(cs[0].validation['rejection_reason'],'stationary_elongated_object')

    def test_size_uses_short_axis_not_trail_length(self):
        cfg=BallValidationConfig()
        self.assertTrue(size_check(12.,12.,cfg)[0]);self.assertFalse(size_check(50.,12.,cfg)[0])


class AdmissionTests(unittest.TestCase):
    def confirmed(self):
        gate=BallTrackGate()
        for i in range(3):
            t=1.+i*.033;p=np.array([0.,0.,2.])
            a=gate.propose(p,t,('rgb',i),t)
            if a.accepted:gate.commit(p,t,t,a.new_track)
        self.assertTrue(gate.valid(t));return gate,t

    def test_persistence_without_identity_never_confirms(self):
        g=BallTrackGate()
        for i in range(20):self.assertFalse(g.propose([0,0,2],i*.01,('ir',i)).accepted)
        self.assertFalse(g.valid(.2))

    def test_repeated_frame_cannot_accumulate_confirmation(self):
        g=BallTrackGate()
        for i in range(10):self.assertFalse(g.propose([0,0,2],1.,('rgb',1),1.).accepted)
        self.assertEqual(len(g.pending),1)

    def test_monotonic_unique_confirmation_then_loss_and_reacquire(self):
        g,t=self.confirmed();old=g.track_id
        g.tick(t+.04);self.assertEqual(g.state,'PREDICTED')
        self.assertFalse(g.tick(t+.2));self.assertEqual(g.state,'LOST')
        self.assertFalse(g.valid(t+.2))
        self.assertTrue(g.tick(t+.31))
        for i in range(3):
            tt=t+.4+i*.03;a=g.propose([.5,0,2],tt,('rgb',20+i),tt)
            if a.accepted:g.commit([.5,0,2],tt,tt,a.new_track)
        self.assertEqual(g.track_id,old+1)

    def test_out_of_order_rgb_does_not_reset_good_track(self):
        g,t=self.confirmed();before=copy.deepcopy(g.position)
        self.assertFalse(g.propose([1,0,2],t-.01,('rgb','old'),t-.01).accepted)
        self.assertTrue(g.valid(t));np.testing.assert_array_equal(g.position,before)

    def test_single_outlier_cannot_change_identity(self):
        g,t=self.confirmed()
        self.assertFalse(g.propose([1,0,2],t+.01,('ir',99),t+.01).accepted)
        np.testing.assert_array_equal(g.position,[0,0,2])

    def test_ir_measurements_cannot_keep_identity_alive_indefinitely(self):
        g,t=self.confirmed()
        for i in range(1,20):
            tt=t+i*.01;a=g.propose([0,0,2],tt,('ir',i))
            if a.accepted:g.commit([0,0,2],tt,None)
        self.assertFalse(g.valid(tt))

    def test_filter_rejection_does_not_change_gate(self):
        s=system();g,t=self.confirmed();s.ball_gate=g
        s.camera_ball_tracker.update(np.array([0,0,2.]),t,.8)
        table=SimpleNamespace(T=np.eye(4))
        previous=s.camera_ball_tracker.filter.x.copy()
        s.camera_ball_tracker.update=Mock(return_value=False)
        accepted=s._admit_ball_measurement(np.array([.02,0,2.]),t+.03,('ir',55),.8,t+.03,.03,table)
        self.assertFalse(accepted)
        np.testing.assert_array_equal(s.camera_ball_tracker.filter.x,previous)
        self.assertEqual(s.ball_gate.last_measurement,t)

    def test_invalid_track_publishes_null_coordinates(self):
        s=system();s.camera_ball_tracker.update(np.array([0,0,2.]),1.,.8)
        s.ball_gate.state='LOST'
        out=s._publish(SimpleNamespace(timestamp_s=1.01,host_timestamp_s=1.,frame_number=1),None,False,'prediction_only',None,{})
        self.assertFalse(out['ball']['valid']);self.assertIsNone(out['ball']['position_camera_m'])

    def test_delayed_rgb_uses_historical_ir_timestamp(self):
        s=system();point=np.array([0.,0.,2.])
        for i,t in enumerate((.94,.97)):
            s.ball_gate.propose(point,t,('rgb',i),t)
        image=np.zeros((240,320),np.uint8)
        past=SimpleNamespace(timestamp_s=1.,frame_number=10,ir_left=image,ir_right=image)
        current=SimpleNamespace(timestamp_s=1.06,frame_number=16,ir_left=image,ir_right=image)
        c=candidate(160,120)
        s.ir_evidence_buffer=[dict(packet=past,left=[c],right=[c],prediction=None,track_id=None)]
        s.rgb_state.set_candidates(1.001,[c],None)
        s.left_ir_detector.detect=Mock(return_value=([c],image))
        s.right_ir_detector.detect=Mock(return_value=([c],image))
        m=app.StereoBallMeasurement(point,c,c,.8,.1,identity_timestamp_s=1.001)
        s.selector.select=Mock(side_effect=[m,None])
        measured,source,_,_=s._process_ball_packet(current,None)
        self.assertTrue(measured);self.assertEqual(source,'ir_stereo_delayed')
        self.assertEqual(s.ball_gate.last_measurement,1.)
        self.assertEqual(s.camera_ball_tracker.filter.last_measurement_timestamp_s,1.)
        self.assertTrue(s.ball_gate.valid(1.06))


class StereoTests(unittest.TestCase):
    def rgb(self,time=1.,center=(160,120)):
        c=candidate(*center);c.validation={'identity_ok':True,'depth_source':'stereo_required'}
        return time,[c]

    def test_good_pair_requires_rgb_identity_to_initialize(self):
        selector=app.StereoCandidateSelector(model(),.02)
        left=[candidate(160,120)];right=[candidate(131.5,120)]
        self.assertIsNone(selector.select(left,right,timestamp_s=1.))
        self.assertIsNotNone(selector.select(left,right,timestamp_s=1.,rgb_observation=self.rgb()))

    def test_wrong_epipolar_size_and_rgb_match_rejected(self):
        selector=app.StereoCandidateSelector(model(),.02)
        self.assertIsNone(selector.select([candidate(160,120)],[candidate(131.5,145)],timestamp_s=1.,rgb_observation=self.rgb()))
        self.assertIsNone(selector.select([candidate(160,120,40)],[candidate(131.5,120,40)],timestamp_s=1.,rgb_observation=self.rgb()))
        self.assertIsNone(selector.select([candidate(160,120)],[candidate(131.5,120)],timestamp_s=1.,rgb_observation=self.rgb(center=(200,180))))

    def test_stale_rgb_cannot_confirm_but_does_not_veto_existing_track(self):
        selector=app.StereoCandidateSelector(model(),.02)
        kwargs=dict(timestamp_s=1.,rgb_observation=self.rgb(.8,center=(200,180)))
        left=[candidate(160,120)];right=[candidate(131.5,120)]
        self.assertIsNone(selector.select(left,right,**kwargs))
        self.assertIsNotNone(selector.select(left,right,tracking=True,**kwargs))

    def test_ambiguous_disparity_is_not_forced_winner(self):
        selector=app.StereoCandidateSelector(model(),.02)
        left=[candidate(160,120)];right=[candidate(131.5,120),candidate(130.,120)]
        self.assertIsNone(selector.select(left,right,timestamp_s=1.,rgb_observation=self.rgb()))
        self.assertEqual(selector.diagnostics['reason'],'ambiguous_stereo')

    def test_dark_ir_texture_does_not_become_stereo_identity(self):
        selector=app.StereoCandidateSelector(model(),.02)
        image=np.full((240,320),17,np.uint8)
        self.assertIsNone(selector.select([candidate(160,120)],[candidate(131.5,120)],
            timestamp_s=1.,rgb_observation=self.rgb(),left_image=image,right_image=image))
        self.assertIn('insufficient_ir_contrast',selector.diagnostics['rejected'])


if __name__=='__main__':unittest.main(verbosity=2)
