import copy
from types import SimpleNamespace
import unittest
import numpy as np
from ball_motion import TimestampedBallFilter,measurement_covariance
from ball_validation import BallTrackGate
from test_ball_validation import system
import test_ball_validation as helpers
import d455_table_tennis_tracker as app


class MotionTests(unittest.TestCase):
    def test_below_plane_sample_does_not_teleport(self):
        f=app.BallKalmanFilter()
        x=np.array([0.,0.,-.017604095,0.,0.,-.5712817])
        after,_=f._predict_once(x,np.eye(6),.011133432)
        self.assertLess(after[2],x[2])
        self.assertLess(abs(after[2]-x[2]),.008)

    def test_delayed_update_matches_chronological_fusion(self):
        ordered=TimestampedBallFilter();late=TimestampedBallFilter()
        times=[1.+i*.02 for i in range(8)]
        def feed(f,i):
            t=times[i];p=np.array([4*(t-1),0.,2.])
            self.assertTrue(f.update(p,t,.8,np.eye(3)*.02**2,('test',i)))
        for i in range(8):feed(ordered,i)
        for i in [0,1,2,4,5,6,3,7]:feed(late,i)
        np.testing.assert_allclose(late.x,ordered.x,atol=1.e-10)
        np.testing.assert_allclose(late.P,ordered.P,atol=1.e-10)
        self.assertEqual(late.counters['late_measurement_fused'],1)

    def test_rejected_late_outlier_cannot_contaminate(self):
        f=TimestampedBallFilter()
        for i in range(8):self.assertTrue(f.update([0,0,2],1+i*.02,.9,key=('test',i)))
        before=f.x.copy();P=f.P.copy();t=f.timestamp_s
        self.assertFalse(f.update([2,3,4],1.07,.9,key=('bad',1)))
        np.testing.assert_array_equal(f.x,before);np.testing.assert_array_equal(f.P,P)
        self.assertEqual(f.timestamp_s,t)

    def test_correlated_sources_are_not_double_counted(self):
        f=TimestampedBallFilter();p=np.array([0,0,2.])
        self.assertTrue(f.update(p,1,.8,np.eye(3)*.04**2,('ir',1),'ir_stereo'))
        self.assertTrue(f.update(p,1.005,.8,np.eye(3)*.02**2,('rgb',1),'rgb_depth_verified'))
        self.assertEqual(len(f.events),1)
        self.assertFalse(f.update(p,1.008,.8,np.eye(3)*.04**2,('ir',2),'ir_stereo'))

    def test_duplicate_and_expired_history_rejected(self):
        f=TimestampedBallFilter()
        for i in range(30):f.update([0,0,2],1+i*.02,.8,key=('ir',i))
        self.assertFalse(f.update([0,0,2],1.58,.8,key=('ir',29)))
        self.assertFalse(f.update([0,0,2],1.1,.8,key=('old',99)))
        self.assertLessEqual(len(f.events),12)
        self.assertIsNone(f.predict_state(.5))

    def test_fast_motion_retains_velocity(self):
        f=TimestampedBallFilter()
        for i in range(20):
            t=i/60.;self.assertTrue(f.update([6*t,0,2.],t,.9,np.eye(3)*.015**2,key=('ir',i)))
        self.assertAlmostEqual(f.x[3],6.,delta=.15)
        self.assertLess(abs(f.predict_state(t+.02)[0][0]-6*(t+.02)),.015)

    def test_ir_depth_has_larger_uncertainty_than_lateral(self):
        R=measurement_covariance([0,0,1.56],'ir_stereo')
        self.assertGreater(R[2,2],R[0,0]*10)

    def test_flight_acceleration_has_bounded_short_prediction_error(self):
        f=TimestampedBallFilter()
        for i in range(30):
            t=i/60.;p=[4*t,.5*9.81*t*t,2.]
            self.assertTrue(f.update(p,t,.9,np.eye(3)*.015**2,key=('ir',i)))
        horizon=.02
        expected=np.array([4*(t+horizon),.5*9.81*(t+horizon)**2,2.])
        self.assertLess(np.linalg.norm(f.predict_state(t+horizon)[0][:3]-expected),.04)

    def test_table_validity_never_switches_camera_estimator(self):
        s=system();g,t=helpers.AdmissionTests().confirmed();s.ball_gate=g
        s.camera_ball_tracker.update(np.array([0,0,2.]),t,.8)
        packet=SimpleNamespace(timestamp_s=t+.01,host_timestamp_s=1.,frame_number=10)
        table=SimpleNamespace(T=np.eye(4),metadata={'table_frame_id':1},last_measurement_timestamp_s=t,confidence=.9)
        a=s._publish(packet,None,False,'prediction_only',None,{})
        s._sync_table_frame(table)
        b=s._publish(packet,table,False,'prediction_only',None,{'table_frame_id':1})
        np.testing.assert_array_equal(a['ball']['position_camera_m'],b['ball']['position_camera_m'])
        self.assertTrue(b['ball']['world_valid'])

    def test_identity_only_does_not_extend_3d_validity(self):
        g,t=helpers.AdmissionTests().confirmed();old_id=g.track_id
        g.refresh_identity(t+.13);g.tick(t+.13)
        self.assertFalse(g.valid(t+.13));self.assertEqual(g.state,'LOST')
        decision=g.propose([0,0,2],t+.15,('rgb',50),t+.15)
        self.assertTrue(decision.accepted);self.assertFalse(decision.new_track)
        g.commit([0,0,2],t+.15,t+.15)
        self.assertEqual(g.track_id,old_id)


if __name__=='__main__':unittest.main(verbosity=2)
