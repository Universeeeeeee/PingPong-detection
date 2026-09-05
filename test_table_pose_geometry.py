"""Geometry and lifecycle regressions; no RealSense device is opened."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pyrealsense2 as rs

from table_pose_geometry import FixedTablePoseTracker, rotation_distance


def tracker():
    intr=rs.intrinsics();intr.width=1280;intr.height=720
    intr.fx=600.;intr.fy=600.;intr.ppx=640.;intr.ppy=360.
    intr.model=rs.distortion.none;intr.coeffs=[0.]*5
    return FixedTablePoseTracker(intr,.001,2.74,1.525,
        [(np.array([85,80,40],np.uint8),np.array([135,255,255],np.uint8))],
        'unused_pose.npz',confirm_frames=3,validation_hz=20.)


def scene(t,shift=0.,width=1.525):
    s,c=np.sin(np.deg2rad(25)),np.cos(np.deg2rad(25))
    T=np.eye(4);T[:3,:3]=np.array([[0.,-1.,0.],[-s,0.,-c],[c,0.,-s]])
    T[:3,3]=[shift,0.,2.5]
    corners=t.corners.copy();corners[:,1]*=width/1.525
    quad=np.rint(t.camera.project(corners@T[:3,:3].T+T[:3,3])).astype(np.int32)
    bgr=np.full((720,1280,3),40,np.uint8)
    cv2.fillConvexPoly(bgr,quad,(170,65,25))
    cv2.polylines(bgr,[quad],True,(235,235,235),5)
    mask=np.zeros((720,1280),np.uint8);cv2.fillConvexPoly(mask,quad,255)
    yy,xx=np.nonzero(mask);rays=t.camera.rays(np.column_stack((xx,yy)))
    n=T[:3,2];d=-n@T[:3,3]
    z=-d/(rays@n)
    noise=np.random.default_rng(4).normal(0,.002,len(z))
    depth=np.zeros(mask.shape,np.uint16);depth[yy,xx]=np.rint((z+noise)*1000).astype(np.uint16)
    return bgr,depth,T


class GeometryTests(unittest.TestCase):
    def test_complete_table(self):
        t=tracker();b,d,gt=scene(t);obs,why=t.observe(b,d);self.assertIsNone(why)
        T,metrics,why=t.detect(obs);self.assertIsNone(why)
        self.assertLess(np.linalg.norm(T[:3,3]-gt[:3,3]),.045)
        self.assertLess(rotation_distance(T[:3,:3],gt[:3,:3]),np.deg2rad(3))

    def test_missing_corner(self):
        t=tracker();b,d,gt=scene(t)
        b[550:, :420]=40;d[550:,:420]=0
        obs,_=t.observe(b,d);T,metrics,why=t.detect(obs)
        self.assertIsNone(why);self.assertLess(np.linalg.norm(T[:3,3]-gt[:3,3]),.06)
        self.assertTrue(any(metrics['corner_inferred']))

    def test_three_sides(self):
        t=tracker();b,d,gt=scene(t)
        # The near short edge is fully occluded; the far short edge and both
        # long sides are visible. Occluder boundary must not become a table end.
        b[540:]=40;d[540:]=0
        obs,_=t.observe(b,d);T,metrics,why=t.detect(obs)
        self.assertIsNone(why);self.assertLess(np.linalg.norm(T[:3,3]-gt[:3,3]),.07)
        self.assertEqual(metrics['observed_edge_count'],3)

    def test_two_parallel_sides_cannot_initialize(self):
        t=tracker();b,d,_=scene(t)
        b[:325]=40;d[:325]=0;b[540:]=40;d[540:]=0
        obs,_=t.observe(b,d);T,_,_=t.detect(obs)
        self.assertIsNone(T)

    def test_wrong_size_and_background_lines(self):
        t=tracker();b,d,_=scene(t,width=1.2)
        cv2.line(b,(10,80),(1100,80),(255,255,255),5)
        obs,_=t.observe(b,d);T,_,_=t.detect(obs)
        self.assertIsNone(T)

    def test_center_line_is_not_outer_edge(self):
        t=tracker();b,d,gt=scene(t)
        q=t.camera.project(np.array([[0,-.7625,0],[0,.7625,0]])@gt[:3,:3].T+gt[:3,3])
        cv2.line(b,tuple(np.rint(q[0]).astype(int)),tuple(np.rint(q[1]).astype(int)),(255,255,255),7)
        obs,_=t.observe(b,d);T,_,why=t.detect(obs)
        self.assertIsNone(why);self.assertLess(np.linalg.norm(T[:3,3]-gt[:3,3]),.05)

    def test_behind_camera_and_wrong_normal_rejected(self):
        t=tracker();b,d,T=scene(t);obs,_=t.observe(b,d)
        behind=T.copy();behind[2,3]=-2
        self.assertEqual(t.validate(behind,obs,True)[1],'behind_camera')
        wrong=T.copy();wrong[:3,1:3]*=-1
        self.assertEqual(t.validate(wrong,obs,True)[1],'plane_pose_disagreement')

    def test_lock_partial_occlusion_loss_recovery_and_new_epoch(self):
        t=tracker();b,d,_=scene(t)
        for k in range(4):t.update(b,d,k*.1)
        snap,meta=t.snapshot(.3);self.assertIsNotNone(snap);locked=snap.T.copy();epoch=meta['table_frame_id']
        partial=b.copy();pd=d.copy();partial[550:,:420]=40;pd[550:,:420]=0
        for k in range(4,7):t.update(partial,pd,k*.1)
        snap,meta=t.snapshot(.6);self.assertIsNotNone(snap)
        np.testing.assert_array_equal(snap.T,locked)
        np.testing.assert_array_equal(snap.linear_velocity,np.zeros(3))
        black=np.zeros_like(b);zero=np.zeros_like(d)
        t.update(black,zero,.8);self.assertIsNone(t.snapshot(.8)[0]);self.assertEqual(t.snapshot(.8)[1]['state'],'HOLDING')
        t.update(black,zero,1.4);self.assertEqual(t.snapshot(1.4)[1]['state'],'LOST')
        t.update(b,d,1.5);self.assertEqual(t.snapshot(1.5)[1]['table_frame_id'],epoch)
        # A new camera-relative position must be independently confirmed and
        # creates a new coordinate-system epoch instead of changing the old one.
        moved,md,_=scene(t,shift=.20)
        for k in range(16,21):t.update(moved,md,k*.1)
        snap,meta=t.snapshot(2.)
        self.assertIsNotNone(snap);self.assertGreater(meta['table_frame_id'],epoch)
        self.assertGreater(np.linalg.norm(snap.T[:3,3]-locked[:3,3]),.12)

    def test_no_measurements_cannot_keep_valid_forever(self):
        t=tracker();b,d,_=scene(t)
        for k in range(4):t.update(b,d,k*.1)
        self.assertIsNone(t.snapshot(2.)[0]);self.assertFalse(t.snapshot(2.)[1]['valid'])

    def test_saved_pose_is_only_an_unverified_candidate(self):
        t=tracker();b,d,T=scene(t)
        with tempfile.TemporaryDirectory() as directory:
            t.pose_file=Path(directory)/'pose.npz'
            np.savez(t.pose_file,T_camera_table=T,table_length=2.74,table_width=1.525)
            self.assertTrue(t.load_initial_pose(0.))
            self.assertIsNone(t.snapshot(0.)[0])
            for k in range(4):t.update(b,d,k*.1)
            self.assertIsNotNone(t.snapshot(.3)[0])
            np.savez(t.pose_file,T_camera_table=T,table_length=1.37,table_width=1.525)
            self.assertFalse(t.load_initial_pose(.4))

    def test_world_invalidity_and_new_epoch_preserve_camera_state(self):
        from d455_table_tennis_tracker import TableTennisPerceptionSystem,BallKalmanFilter,CameraBallTracker
        s=TableTennisPerceptionSystem.__new__(TableTennisPerceptionSystem)
        s.camera_ball_tracker=CameraBallTracker()
        from ball_validation import BallTrackGate
        s.ball_gate=BallTrackGate();s.ball_gate.commit(np.array([0.,0.,2.]),1.,1.,True)
        s.last_admission_reason='accepted_measurement';s.last_ball_payload=None
        s.selector=SimpleNamespace(diagnostics={})
        s.rgb_state=SimpleNamespace(get_candidates=lambda:(None,[]))
        s._ball_table_frame_id=1
        s.camera_ball_tracker.update(np.array([0.,0.,2.]),1.,.8)
        s.stats=SimpleNamespace(latest_fps=90.)
        s.model=SimpleNamespace(selected_ir_fps=90)
        s.capture=SimpleNamespace(hardware_ir_drops=0,callback_errors=0,stereo_queue=SimpleNamespace(dropped=0),color_queue=SimpleNamespace(dropped=0))
        s.pub_socket=Mock();s.jsonl_file=None
        packet=SimpleNamespace(timestamp_s=1.01,host_timestamp_s=1.,frame_number=1)
        out=s._publish(packet,None,True,'ir_stereo',None,{'state':'LOST','table_frame_id':1})
        self.assertFalse(out['ball']['world_valid']);self.assertIsNone(out['ball']['position_table_m'])
        self.assertTrue(out['ball']['camera_valid'])
        before=s.camera_ball_tracker.filter.x.copy()
        s._sync_table_frame(SimpleNamespace(metadata={'table_frame_id':2}))
        self.assertTrue(s.camera_ball_tracker.initialized);self.assertEqual(s._ball_table_frame_id,2)
        np.testing.assert_array_equal(s.camera_ball_tracker.filter.x,before)


if __name__=='__main__':unittest.main(verbosity=2)
