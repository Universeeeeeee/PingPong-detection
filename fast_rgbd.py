"""Experimental per-frame RGB + independently matched IR stereo measurements.

No prediction is exposed as measured depth. RGB and depth timestamps are kept
separate because these D455 streams have different acquisition rates.
"""
import time
from collections import deque
import numpy as np
from ball_image_roi import FastImageBallDetector
from ball_epipolar import EpipolarBallMatcher


class FastRGBDProcessor:
    def __init__(self,model,stereo=None):
        self.model=model;self.image=FastImageBallDetector();self.stereo=stereo or EpipolarBallMatcher(model)
        self.last_depth=None
        self.used_ir=deque(maxlen=128)

    def process(self,bgr,t,available_ir,deadline=None):
        if deadline is not None and time.perf_counter()>=deadline:
            return dict(measured=False,xyz=None,rgb_uv=None,measurement_t=t,image_mode='skipped',image_ms=0.,
                result_2d=dict(valid=False,uv=None,reason='deadline_expired'),source='none',stereo_t=None,
                reason='deadline_expired_before_processing')
        begin=time.perf_counter();result,_,_=self.image.detect(bgr,t)
        image_ms=1000*(time.perf_counter()-begin)
        output=dict(measured=False,xyz=None,rgb_uv=result['uv'],measurement_t=t,rgb_timestamp_s=t,
            result_2d=result,image_mode=self.image.mode,image_ms=image_ms,
            reason='no_image_detection',source='none',stereo_t=None)
        if not result['valid']:return output
        candidate=self.image.selected_candidate;velocity=np.zeros(2)
        if len(self.image.history)>=2:
            a,b=list(self.image.history)[-2:];velocity=(b[1]-a[1])/max(.001,b[0]-a[0])
        if callable(available_ir):available_ir=available_ir()
        pairs=[p for p in available_ir if abs(p['t']-t)<=.018 and abs(p['right_t']-p['t'])<=.001
               and p['t'] not in self.used_ir and (self.last_depth is None or p['t']>self.last_depth[0])]
        def pair_deadline(pair):
            return None if deadline is None else deadline+min(0.,pair['t']-t)
        matches=[];attempts=[];begin=time.perf_counter()
        bounds=(.3,6.)
        if self.last_depth is not None:
            old_t,old_point,old_id=self.last_depth
            if old_id==result['track_id'] and 0<t-old_t<.15:
                margin=.25+20*(t-old_t)
                bounds=(max(.3,float(old_point[2])-margin),min(6.,float(old_point[2])+margin))
        for pair in sorted(pairs,key=lambda p:abs(p['t']-t)):
            if deadline is not None and time.perf_counter()>=deadline:break
            shift=velocity*(pair['t']-t);target=dict(result);c=dict(candidate)
            target['uv']=(np.array(result['uv'])+shift).tolist()
            c['bbox']=(np.array(candidate['bbox'])+np.r_[shift,[0,0]]).tolist()
            m=self.stereo.match(pair['ir_left'],pair['ir_right'],target,c,pair['t'],t,bounds,pair_deadline(pair))
            attempts.append(dict(frame=pair['name'],**self.stereo.diagnostics))
            if m is not None:matches.append(m)
        if not matches and bounds!=(.3,6.):
            for pair in sorted(pairs,key=lambda p:abs(p['t']-t)):
                if deadline is not None and time.perf_counter()>=deadline:break
                shift=velocity*(pair['t']-t);target=dict(result);c=dict(candidate)
                target['uv']=(np.array(result['uv'])+shift).tolist()
                c['bbox']=(np.array(candidate['bbox'])+np.r_[shift,[0,0]]).tolist()
                m=self.stereo.match(pair['ir_left'],pair['ir_right'],target,c,pair['t'],t,deadline=pair_deadline(pair))
                attempts.append(dict(frame=pair['name'],full_depth_search=True,**self.stereo.diagnostics))
                if m is not None:matches.append(m)
        if not matches:
            # At a bounce, constant image velocity can point in the wrong
            # direction. Re-search the actual observed RGB region, still using
            # only delivered current IR images and the same geometric checks.
            for pair in sorted(pairs,key=lambda p:abs(p['t']-t)):
                if deadline is not None and time.perf_counter()>=deadline:break
                if np.linalg.norm(velocity*(pair['t']-t))<2:continue
                m=self.stereo.match(pair['ir_left'],pair['ir_right'],result,candidate,pair['t'],t,deadline=pair_deadline(pair))
                attempts.append(dict(frame=pair['name'],unshifted=True,**self.stereo.diagnostics))
                if m is not None:matches.append(m)
        output.update(stereo_ms=1000*(time.perf_counter()-begin),attempts=attempts,
            available_pairs=len(pairs),reason='no_verified_stereo')
        if not matches:return output
        matches.sort(key=lambda m:m['cost']);best=matches[0]
        # All three coordinates come from this observed stereo pair. RGB is
        # association evidence at a different time, never a replacement ray.
        xyz=np.asarray(best['xyz_ir'],dtype=float)
        measurement_t=best['ir_t']
        if self.last_depth is not None:
            old_t,old_point,old_id=self.last_depth
            if old_id==result['track_id'] and 0<measurement_t-old_t<.15 and np.linalg.norm(xyz-old_point)>20*(measurement_t-old_t)+.05:
                output['reason']='impossible_3d_displacement';return output
        self.last_depth=(measurement_t,xyz.copy(),result['track_id'])
        self.used_ir.append(best['ir_t'])
        output.update(measured=True,xyz=xyz.tolist(),source='ir_stereo',measurement_t=measurement_t,
            coordinate_frame='d455_left_ir_optical_frame',
            stereo_right_t=next(p['right_t'] for p in pairs if p['t']==measurement_t),
            rgb_projection_at_measurement_t=best['rgb_projection'],
            stereo_t=best['ir_t'],depth_skew_ms=(best['ir_t']-t)*1000,
            depth_sample_new=True,
            stereo_measurement=best,reason='verified_stereo')
        return output
