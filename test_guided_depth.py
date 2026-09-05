import unittest
import cv2
import numpy as np
from unittest.mock import Mock
from types import SimpleNamespace
from test_ball_validation import model,system,candidate
from ball_validation import contour_features,appearance_check,BallValidationConfig
import d455_table_tennis_tracker as app


class GuidedDepthTests(unittest.TestCase):
    def scene(self):
        m=model();c=candidate(160,120)
        c.validation=dict(identity_ok=True,depth_source='stereo_required')
        l=np.full((240,320),20,np.uint8);r=l.copy()
        cv2.circle(l,(160,120),6,100,-1);cv2.circle(r,(132,120),6,100,-1)
        return m,c,l,r

    def test_recovers_stereo_without_any_depth_map_or_previous_frame(self):
        m,c,l,r=self.scene()
        lc=app.rgb_guided_ir_candidates(l,[c],m);rc=app.rgb_guided_ir_candidates(r,[c],m,True)
        b=app.StereoCandidateSelector(m,.02).select(lc,rc,timestamp_s=1.,
            rgb_observation=(1.,[c]),left_image=l,right_image=r)
        self.assertIsNotNone(b)
        self.assertAlmostEqual(b.position_left_camera[2],600*.095/28,places=4)

    def test_dark_frame_and_ineligible_color_cannot_create_candidates(self):
        m,c,l,r=self.scene()
        self.assertEqual(app.rgb_guided_ir_candidates(np.full_like(l,16),[c],m),[])
        c.validation['identity_ok']=False
        self.assertEqual(app.rgb_guided_ir_candidates(l,[c],m),[])

    def test_straight_bright_background_edge_is_not_an_independent_blob(self):
        m,c,l,r=self.scene();l[:]=20;cv2.line(l,(0,120),(319,120),150,3)
        self.assertEqual(app.rgb_guided_ir_candidates(l,[c],m),[])

    def test_delayed_rgb_researches_raw_ir_outside_old_empty_roi(self):
        m,c,l,r=self.scene();s=system()
        for i,t in enumerate((.94,.97)):
            s.ball_gate.propose([0,0,600*.095/28],t,('rgb',i),t)
        old=SimpleNamespace(timestamp_s=1.,frame_number=10,ir_left=l,ir_right=r)
        dark=np.full_like(l,16)
        now=SimpleNamespace(timestamp_s=1.06,frame_number=16,ir_left=dark,ir_right=dark)
        s.ir_evidence_buffer=[dict(packet=old,left=[],right=[],prediction=None,track_id=None)]
        s.rgb_state.set_candidates(1.001,[c],None)
        s.left_ir_detector.detect=Mock(return_value=([],dark));s.right_ir_detector.detect=Mock(return_value=([],dark))
        measured,source,_,_=s._process_ball_packet(now,None)
        self.assertTrue(measured);self.assertEqual(source,'ir_stereo_delayed')
        self.assertEqual(s.ball_gate.last_measurement,1.)

    def test_moderate_motion_blur_has_no_aspect_classification_gap(self):
        image=np.zeros((80,80,3),np.uint8);mask=np.zeros((80,80),np.uint8)
        cv2.ellipse(mask,(40,40),(10,6),0,0,360,255,-1);image[mask>0]=(0,140,255)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        f=contour_features(image,contours[0],[(np.array([5,60,80]),np.array([35,255,255]))])
        self.assertGreater(f['aspect'],1.6);self.assertLess(f['aspect'],1.8)
        f['motion_ratio']=1.;f['edge_support']=1.
        self.assertTrue(appearance_check(f,BallValidationConfig())[0])
        f['motion_ratio']=0.
        self.assertFalse(appearance_check(f,BallValidationConfig())[0])


if __name__=='__main__':unittest.main()
