"""Independent ball evidence and track admission. No camera or GUI is opened here.

Scores rank candidates; they are not calibrated probabilities. Depth obtained by
assuming a ball diameter is NEVER used as independent size/identity evidence.
"""
from collections import deque
from dataclasses import dataclass, field
import math

import cv2
import numpy as np


@dataclass
class BallValidationConfig:
    min_width_px: float = 4.0
    min_area_px: float = 12.0
    size_relative_tolerance: float = .40
    size_pixel_tolerance: float = 2.0
    rgb_match_seconds: float = .045
    identity_hold_seconds: float = .15
    predict_seconds: float = .12
    confirm_observations: int = 3
    confirm_window_seconds: float = .18
    max_speed_mps: float = 20.0
    ambiguity_margin: float = .18
    reassociate_seconds: float = .30


def contour_features(bgr, contour, hsv_ranges):
    """Measure an actual candidate boundary plus a wider, less saturated context.

    findContours returns closed polygons even for fragments. 'edge_support' thus
    measures support in the original image, not the polygon's formal closure.
    """
    h, w = bgr.shape[:2]
    x,y,bw,bh = cv2.boundingRect(contour)
    rect = cv2.minAreaRect(contour)
    minor, major = sorted([float(rect[1][0])+1., float(rect[1][1])+1.])
    pad = max(6, int(minor*.8))
    x0,y0,x1,y1 = max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad)
    patch = bgr[y0:y1,x0:x1]
    local = contour - np.array([[[x0,y0]]],dtype=contour.dtype)
    inside = np.zeros(patch.shape[:2],np.uint8)
    cv2.drawContours(inside,[local],-1,255,-1)
    # Context includes nearby red/orange hues, which the narrow orange mask may
    # split into tiny fragments on a red basket. White uses low saturation.
    hsv = cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    context = np.zeros(inside.shape,np.uint8)
    for low,high in hsv_ranges:
        lo,hi = np.array(low,dtype=int).copy(),np.array(high,dtype=int).copy()
        lo[0]=max(0,lo[0]-10);hi[0]=min(179,hi[0]+10)
        lo[1]=max(0,lo[1]-25);lo[2]=max(0,lo[2]-20)
        context |= cv2.inRange(hsv,lo.astype(np.uint8),hi.astype(np.uint8))
        if int(low[0]) <= 10 and int(low[1]) > 30:
            context |= cv2.inRange(hsv,np.array([170,lo[1],lo[2]],np.uint8),np.array([179,255,255],np.uint8))
    distance = cv2.distanceTransform(255-inside,cv2.DIST_L2,3)
    ring = (distance >= 2.) & (distance <= max(4.,minor*.6))
    surrounding = float(np.mean(context[ring]>0)) if ring.any() else 1.
    gray = cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,35,100)
    edge_distance = cv2.distanceTransform(255-edges,cv2.DIST_L2,3)
    outline = np.zeros_like(inside);cv2.drawContours(outline,[local],-1,255,1)
    support = float(np.mean(edge_distance[outline>0] <= 2.)) if np.any(outline) else 0.
    area = float(cv2.contourArea(contour))
    perimeter = cv2.arcLength(contour,True)
    circularity = min(1.,4.*math.pi*area/max(perimeter*perimeter,1.))
    hull = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area/max(hull,1.)
    aspect = major/max(minor,1.)
    return dict(minor_px=minor,major_px=major,area_px=area,circularity=circularity,
                solidity=solidity,aspect=aspect,edge_support=support,
                surrounding_color=surrounding,
                clipped=bool(x<=1 or y<=1 or x+bw>=w-1 or y+bh>=h-1),
                mode='streak' if aspect>1.6 else 'compact')


def appearance_check(features,cfg):
    f=features
    if f['clipped']:
        return False,'partial_boundary'
    if f['minor_px'] < cfg.min_width_px or f['area_px'] < cfg.min_area_px:
        return False,'too_few_pixels'
    if f['edge_support'] < .45:
        return False,'unsupported_boundary'
    if f['mode']=='compact':
        if f['circularity'] < .72 or f['solidity'] < .88 or f['aspect'] > 1.6:
            return False,'irregular_shape'
    else:
        if f['solidity'] < .80 or f['aspect'] > 7.:
            return False,'irregular_streak'
        if f.get('motion_ratio',0.) < .08:
            return False,'stationary_elongated_object'
    # Context alone does not reject. Independent foreground depth can resolve it.
    return True,'embedded_color' if f['surrounding_color'] > .40 else 'appearance_ok'


def depth_evidence(depth,scale,contour):
    """Robust foreground depth, with erosion to avoid mixed aligned-depth edges."""
    h,w=depth.shape
    x,y,bw,bh=cv2.boundingRect(contour)
    pad=8;x0,y0,x1,y1=max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad)
    mask=np.zeros((y1-y0,x1-x0),np.uint8)
    local=contour-np.array([[[x0,y0]]],dtype=contour.dtype)
    cv2.drawContours(mask,[local],-1,255,-1)
    core=cv2.erode(mask,np.ones((3,3),np.uint8))>0
    values=depth[y0:y1,x0:x1].astype(float)*scale
    valid=(values>.2)&(values<6.)
    samples=values[core&valid]
    fraction=float(len(samples)/max(1,int(core.sum())))
    result=dict(reliable=False,z_m=None,valid_fraction=fraction,mad_m=None,
                foreground_separation_m=None,reason='depth_missing')
    if len(samples)<8 or fraction<.65:
        return result
    z=float(np.median(samples));mad=float(np.median(np.abs(samples-z)))
    spread=float(np.percentile(samples,90)-np.percentile(samples,10))
    result.update(z_m=z,mad_m=mad,spread_m=spread)
    if mad>max(.012,.008*z) or spread>max(.06,.035*z):
        result['reason']='mixed_depth';return result
    distance=cv2.distanceTransform(255-mask,cv2.DIST_L2,3)
    ring=(distance>=3.)&(distance<=7.)&valid
    if int(ring.sum())>=12:
        separation=float(np.median(values[ring])-z)
        result['foreground_separation_m']=separation
        # A uniform background surface may produce a very clean patch too.
        if separation < .008:
            result['reason']='background_depth';return result
    else:
        result['reason']='depth_context_missing';return result
    result.update(reliable=True,reason='depth_ok')
    return result


def size_check(observed_px,expected_px,cfg,depth_sigma_m=0.,z_m=1.):
    if not np.isfinite([observed_px,expected_px,z_m]).all() or expected_px<=0 or z_m<=0:
        return False,float('inf')
    tolerance=cfg.size_pixel_tolerance+cfg.size_relative_tolerance*expected_px
    tolerance += min(expected_px*.25,2.*expected_px*depth_sigma_m/z_m)
    error=abs(observed_px-expected_px)/max(tolerance,1.)
    return error<=1.,float(error)


def validate_rgb_candidate(candidate,bgr,depth,scale,fx,ball_radius,cfg,hsv_ranges):
    f=candidate.features
    ok,reason=appearance_check(f,cfg)
    d=depth_evidence(depth,scale,candidate.contour)
    diagnostics=dict(appearance=dict(f),depth=d,rejection_reason=None,identity_ok=False,
                     depth_source='unavailable',size_error=None)
    if not ok:
        diagnostics['rejection_reason']=reason;return diagnostics
    if d['reliable']:
        # D455 measures the visible surface; a radius-scale centre correction is
        # bounded by the size/depth tolerance rather than fitting the diameter.
        z=d['z_m']+ball_radius
        match,error=size_check(f['minor_px'],fx*(2.*ball_radius)/z,cfg,d['mad_m'],z)
        diagnostics.update(size_error=error,depth_source='aligned_depth',z_m=z)
        if not match:
            diagnostics['rejection_reason']='physical_size';return diagnostics
    if reason=='embedded_color' and not (d['reliable'] and d['foreground_separation_m']>.025 and f['edge_support']>.7):
        diagnostics['rejection_reason']='embedded_color_unresolved';return diagnostics
    diagnostics['identity_ok']=True
    diagnostics['depth_source']='aligned_depth' if d['reliable'] else 'stereo_required'
    return diagnostics


@dataclass
class Admission:
    accepted: bool
    new_track: bool = False
    reason: str = ''


class BallTrackGate:
    """Temporal confirmation cannot replace independent image/geometry evidence.

    Only commit after every downstream filter accepts (caller uses a transaction).
    All times refer to actual sensor measurements, not their processing time.
    """
    def __init__(self,cfg=None):
        self.cfg=cfg or BallValidationConfig()
        self.state='SEARCHING';self.reason='no_confirmed_ball';self.track_id=0
        self.last_measurement=None;self.last_identity=None
        self.pending=deque();self.seen=deque(maxlen=128)
        self.position=None;self.velocity=np.zeros(3)
        self.history=deque(maxlen=64);self.stationary=False
        self.measurement_sigma=.025
        self._still_since=None

    def reset(self):
        old_id=self.track_id
        self.__init__(self.cfg);self.track_id=old_id

    def valid(self,t):
        return (self.last_measurement is not None and self.last_identity is not None
                and 0<=t-self.last_measurement<=self.cfg.predict_seconds
                and 0<=t-self.last_identity<=self.cfg.identity_hold_seconds
                and self.state in ('CONFIRMED','PREDICTED'))

    def tick(self,t):
        if self.last_measurement is not None and not self.valid(t):
            self.state='LOST';self.reason='measurement_or_identity_expired'
            # Retain a bounded association hypothesis, never a valid 3D output.
            if t-self.last_measurement > self.cfg.reassociate_seconds:
                self.last_measurement=None;self.last_identity=None
                self.position=None;self.velocity[:]=0;self.pending.clear()
                self.history.clear();self.stationary=False;self._still_since=None
                return True
            return False
        while self.pending and t-self.pending[0][0]>self.cfg.confirm_window_seconds:
            self.pending.popleft()
        if self.last_measurement is not None and t>self.last_measurement:
            self.state='PREDICTED'
        elif not self.pending and self.state=='CANDIDATE':
            self.state='SEARCHING'
        return False

    def can_associate(self,t):
        return (self.last_measurement is not None and
                0 <= t-self.last_measurement <= self.cfg.reassociate_seconds)

    def refresh_identity(self,t):
        if self.last_measurement is not None:
            self.last_identity=max(float(t),self.last_identity or float(t))

    def propose(self,p,t,key,identity_time=None,sigma=.025,predicted_position=None,predicted_cov=None):
        p=np.asarray(p,dtype=float)
        if self.last_measurement is not None and t<=self.last_measurement:
            return Admission(False,reason='duplicate_or_out_of_order')
        self.tick(t)
        if key in self.seen:
            return Admission(False,reason='duplicate_measurement')
        self.seen.append(key)
        if not np.isfinite(p).all() or (self.last_measurement is not None and t<=self.last_measurement):
            return Admission(False,reason='invalid_or_out_of_order')
        self.measurement_sigma=float(sigma)
        strong=(identity_time is not None and abs(t-identity_time)<=self.cfg.rgb_match_seconds)
        if self.last_measurement is not None:
            if self.state=='LOST' and not (strong or
                    (self.last_identity is not None and 0<=t-self.last_identity<=self.cfg.rgb_match_seconds)):
                return Admission(False,reason='reassociation_identity_required')
            dt=t-self.last_measurement
            predicted=(self.position+self.velocity*dt if predicted_position is None else predicted_position)
            tolerance=min(.18,.03+2.*sigma+2.*dt)
            if np.linalg.norm(p-predicted)>tolerance:
                self.reason='trajectory_mismatch'
                return Admission(False,reason=self.reason)
            return Admission(True,reason='tracking_evidence')
        if not strong:
            self.reason='independent_identity_required'
            return Admission(False,reason=self.reason)
        if self.pending:
            t0,p0=self.pending[-1]
            if t<=t0:
                return Admission(False,reason='nonindependent_timestamp')
            if t-t0<.012:
                return Admission(False,reason='correlated_confirmation')
            dt=t-t0
            if np.linalg.norm(p-p0)>.05+self.cfg.max_speed_mps*dt:
                self.pending.clear()
            elif len(self.pending)>=2:
                ta,pa=self.pending[-2]
                velocity=(p0-pa)/max(t0-ta,.001)
                if np.linalg.norm(p-(p0+velocity*dt))>.10+3.*sigma+30.*dt*dt:
                    self.pending.clear()
        self.pending.append((t,p.copy()))
        self.state='CANDIDATE';self.reason='confirming_independent_observations'
        if len(self.pending)<self.cfg.confirm_observations:
            return Admission(False,reason=self.reason)
        return Admission(True,new_track=True,reason='independently_confirmed')

    def commit(self,p,t,identity_time,new_track=False,estimated_velocity=None):
        p=np.asarray(p,dtype=float)
        if new_track:
            self.track_id+=1
            self.velocity[:]=0.
        # Use the authoritative estimator; never differentiate asynchronous raw
        # measurements into a second, noisy velocity estimate.
        self.velocity=(np.zeros(3) if estimated_velocity is None else np.asarray(estimated_velocity).copy())
        self.position=p.copy();self.last_measurement=t
        if new_track:self.history.clear()
        self.history.append((t,p.copy()))
        while self.history and t-self.history[0][0]>.25:self.history.popleft()
        still=False
        if len(self.history)>=5 and t-self.history[0][0]>=.15:
            points=np.array([item[1] for item in self.history])
            centre=np.median(points,axis=0)
            tolerance=min(.08,max(.025,2.*self.measurement_sigma))
            still=bool(np.max(np.linalg.norm(points-centre,axis=1))<tolerance and np.linalg.norm(self.velocity)<.15)
        if still:
            if self._still_since is None:self._still_since=t
            if t-self._still_since>=.20:self.stationary=True
        else:
            self._still_since=None
            if np.linalg.norm(self.velocity)>.25 or new_track:self.stationary=False
        if identity_time is not None:
            self.last_identity=max(identity_time,self.last_identity or identity_time)
        self.pending.clear();self.state='CONFIRMED';self.reason='accepted_measurement'

    def metadata(self,t):
        return dict(state=self.state,track_id=self.track_id,valid=self.valid(t),
                    reason=self.reason,pending_observations=len(self.pending),
                    stationary=self.stationary,
                    measurement_timestamp_s=self.last_measurement,
                    last_measurement_camera_m=None if self.position is None else self.position.tolist(),
                    identity_age_s=None if self.last_identity is None else max(0.,t-self.last_identity))
