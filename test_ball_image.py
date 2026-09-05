import json
import unittest
import cv2
import numpy as np
from ball_image import ImageBallDetector


def detector():
    return ImageBallDetector([(np.array([2,75,65],np.uint8),np.array([35,255,255],np.uint8))],True)


def scene(centres=()):
    b=np.full((160,240,3),(100,35,20),np.uint8)
    for xy in centres:cv2.circle(b,xy,8,(15,110,245),-1)
    return b


class ImageTests(unittest.TestCase):
    def test_static_without_depth(self):
        d=detector();r,_,_=d.detect(scene([(100,70)]),1.)
        self.assertTrue(r['valid']);self.assertFalse(r['depth_valid'])
        np.testing.assert_allclose(r['uv'],[100,70],atol=1)
        json.dumps(r)

    def test_missing_ball_is_not_prediction(self):
        d=detector();d.detect(scene([(100,70)]),1.)
        r,_,_=d.detect(scene(),1.033)
        self.assertFalse(r['valid']);self.assertIsNone(r['uv']);self.assertFalse(r['measured'])

    def test_two_equal_objects_are_ambiguous(self):
        r,_,_=detector().detect(scene([(60,70),(170,70)]),1.)
        self.assertFalse(r['valid']);self.assertEqual(r['reason'],'ambiguous_image_candidates')

    def test_red_container_does_not_create_ball(self):
        b=scene();cv2.rectangle(b,(50,30),(180,110),(0,0,230),-1)
        for x in range(55,180,12):cv2.line(b,(x,40),(x,100),(20,40,180),3)
        r,_,_=detector().detect(b,1.)
        self.assertFalse(r['valid'])

    def test_duplicate_frame_does_not_change_history(self):
        d=detector();d.detect(scene([(100,70)]),1.);bg=d.background.copy()
        r,_,_=d.detect(scene([(170,70)]),1.)
        self.assertFalse(r['valid']);np.testing.assert_array_equal(bg,d.background)
        self.assertEqual(len(d.history),1)

    def test_resolution_change_resets_background(self):
        d=detector();d.detect(scene([(100,70)]),1.)
        r,_,_=d.detect(np.zeros((120,160,3),np.uint8),2.)
        self.assertFalse(r['valid']);self.assertEqual(d.background.shape,(120,160,3))

    def test_motion_uses_current_pixels(self):
        d=detector()
        for i,x in enumerate([70,82,94,106,118]):
            r,_,_=d.detect(scene([(x,70)]),1+i/30)
            self.assertTrue(r['valid']);np.testing.assert_allclose(r['uv'],[x,70],atol=1.)

if __name__=='__main__':unittest.main()
