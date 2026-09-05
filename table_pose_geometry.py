"""Fixed-camera table registration from metric outer-boundary evidence.

RGB proposes boundaries; depth fixes the plane. Only three in-plane pose
parameters are fitted. Missing corners and three-sided initial observations
are supported; image clipping and convex-hull completion are never evidence.
"""
from __future__ import annotations

import math
import threading
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def angle_between(a, b):
    return float(np.arccos(np.clip(np.dot(a, b), -1., 1.)))


def rotation_distance(a, b):
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1., 1.)))


@dataclass
class TableSnapshot:
    T: np.ndarray
    timestamp_s: float
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    confidence: float
    last_measurement_timestamp_s: float
    metadata: dict = field(default_factory=dict)


class CameraGeometry:
    def __init__(self, intr):
        self.intr = intr
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.]], float)
        self.dist = np.array(intr.coeffs, float)
        # Verify the actual device model instead of assuming coefficient semantics.
        pixels = np.array([(x, y) for y in np.linspace(2, intr.height-3, 7)
                           for x in np.linspace(2, intr.width-3, 9)])
        rays = self.rays(pixels)
        sdk = np.array([rs.rs2_project_point_to_pixel(intr, p.tolist()) for p in rays])
        self.sdk_error_px = float(np.max(np.linalg.norm(sdk-pixels, axis=1)))
        if self.sdk_error_px > .25:
            raise ValueError(f"RGB distortion model mismatch: {self.sdk_error_px:.3f}px")

    def rays(self, uv):
        uv = np.asarray(uv, float).reshape(-1, 1, 2)
        xy = cv2.undistortPoints(uv, self.K, self.dist).reshape(-1, 2)
        return np.column_stack((xy, np.ones(len(xy))))

    def project(self, xyz):
        xyz = np.asarray(xyz, float).reshape(-1, 3)
        uv, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), self.K, self.dist)
        return uv.reshape(-1, 2)

    def intersect(self, uv, normal, d):
        rays = self.rays(uv)
        den = rays @ normal
        z = np.divide(-d, den, out=np.full(len(den), np.nan), where=np.abs(den) > .04)
        z[(z < .2) | (z > 8)] = np.nan
        return rays * z[:, None]


class FixedTablePoseTracker:
    """Registration, partial-view verification, and atomic validity snapshots."""
    def __init__(self, color_intr, depth_scale, table_length_m, table_width_m,
                 hsv_ranges, pose_file, min_area=15000, confirm_frames=5,
                 hold_seconds=.6, edge_tolerance_px=5., validation_hz=12.):
        self.camera = CameraGeometry(color_intr)
        self.color_intr = color_intr
        self.depth_scale = depth_scale
        self.length, self.width = table_length_m, table_width_m
        self.hsv_ranges = hsv_ranges
        self.pose_file = Path(pose_file)
        self.min_area = min_area or 5000
        self.confirm_frames = confirm_frames
        self.hold_seconds = hold_seconds
        self.edge_tolerance = edge_tolerance_px
        self.validation_hz = validation_hz
        self._lock = threading.RLock()
        self._T = None
        self._last_verified = None
        self._last_attempt = None
        self._pending = []
        self._state = "SEARCHING"
        self._reason = "not_initialized"
        self._metrics = {}
        self._epoch = 0
        self._debug = None
        self._force = False
        self._loaded_candidate = None
        self._rng = np.random.default_rng(741)
        self.last_observation = None
        self.last_candidates = []
        self.corners = np.array([[-self.length/2, -self.width/2, 0],
                                 [self.length/2, -self.width/2, 0],
                                 [self.length/2, self.width/2, 0],
                                 [-self.length/2, self.width/2, 0]], float)
        self._edge_samples = [np.linspace(self.corners[i], self.corners[(i+1)%4], 65)
                              for i in range(4)]

    def _mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        raw = np.zeros(bgr.shape[:2], np.uint8)
        for lo, hi in self.hsv_ranges:
            raw |= cv2.inRange(hsv, lo, hi)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
        keep = np.zeros(count, bool)
        # Both table halves can be disconnected by the net. Keep substantial
        # components, then use the metric plane/rectangle to reject background.
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(1200, self.min_area*.08)
        return (keep[labels]*255).astype(np.uint8)

    def _plane(self, depth, mask):
        interior = cv2.erode(mask, np.ones((11,11), np.uint8))
        yy, xx = np.nonzero(interior[::5, ::5])
        xx, yy = xx*5, yy*5
        z = depth[yy, xx].astype(float)*self.depth_scale
        valid = (z>.3) & (z<6.)
        uv = np.column_stack((xx[valid], yy[valid]))
        pts = self.camera.rays(uv)*z[valid, None]
        if len(pts)<200:
            return None, "insufficient_depth"
        idx = self._rng.permutation(len(pts))[:2400]
        pts, uv = pts[idx], uv[idx]
        train, held = pts[::2], pts[1::2]
        best = None
        best_count = 0
        for _ in range(60):
            a,b,c = train[self._rng.choice(len(train),3,replace=False)]
            n = np.cross(b-a,c-a)
            size = np.linalg.norm(n)
            if size<1e-6: continue
            n /= size
            good = np.abs((train-a)@n)<.018
            if good.sum()>best_count: best, best_count = good, int(good.sum())
        if best is None or best_count < max(120, .5*len(train)):
            return None, "no_dominant_plane"
        fit = train[best]
        center = np.mean(fit,axis=0)
        _,_,vt = np.linalg.svd(fit-center,full_matrices=False)
        n=vt[-1]
        if n@(-center)<0:n=-n
        d=-float(n@center)
        residual = np.abs(held@n+d)
        good = residual<.025
        if np.mean(good)<.60:
            return None,"depth_validation_failed"
        inliers = pts[np.abs(pts@n+d)<.025]
        # Spatial support must extend in two directions, not only a tiny patch.
        singular = np.linalg.svd(inliers-inliers.mean(axis=0),compute_uv=False)/np.sqrt(len(inliers))
        if singular[1]<.12:
            return None,"depth_coverage_too_small"
        q = np.array([0.,0.,1.])-n[2]*n
        if np.linalg.norm(q)<.1:q=np.array([1.,0.,0.])-n[0]*n
        q/=np.linalg.norm(q)
        basis=np.column_stack((q,np.cross(n,q)))
        return dict(n=n,d=d,basis=basis,origin=-d*n,points=inliers,
                    held=held,plane_error_m=float(np.median(residual[good])),
                    plane_inlier_ratio=float(np.mean(good))),None

    def observe(self,bgr,depth):
        mask=self._mask(bgr)
        if np.count_nonzero(mask)<self.min_area:return None,"insufficient_blue_area"
        plane,reason=self._plane(depth,mask)
        if plane is None:return None,reason
        # Remove valid non-planar blue pixels, but keep holes with unknown depth
        # out of plane validation. Preserve the physical silhouette proposal.
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        outline=np.zeros_like(mask)
        cv2.drawContours(outline,contours,-1,255,1)
        near=cv2.dilate(outline,np.ones((15,15),np.uint8))
        edge=cv2.Canny(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),55,140)&near
        edge[:5]=0;edge[-5:]=0;edge[:,:5]=0;edge[:,-5:]=0
        distance=cv2.distanceTransform(255-edge,cv2.DIST_L2,3)
        lines=cv2.HoughLinesP(edge,1,np.pi/720,threshold=50,minLineLength=65,maxLineGap=18)
        metric_lines=[]
        if lines is not None:
            for line in sorted(lines[:,0],key=lambda l:-np.linalg.norm(l[2:]-l[:2]))[:45]:
                xyz=self.camera.intersect(line.reshape(2,2),plane['n'],plane['d'])
                if not np.all(np.isfinite(xyz)):continue
                xy=(xyz-plane['origin'])@plane['basis']
                delta=xy[1]-xy[0];length=np.linalg.norm(delta)
                if .18<length<4:
                    metric_lines.append((xy.mean(axis=0),delta/length,float(length)))
        cloud=(plane['points']-plane['origin'])@plane['basis']
        return dict(plane=plane,mask=mask,edge=edge,distance=distance,lines=metric_lines,
                    pixel_lines=[] if lines is None else lines[:,0].astype(float),
                    cloud=cloud,bgr=bgr),None

    def _pose(self,theta,center,plane):
        a=np.array([np.cos(theta),np.sin(theta)])
        b=np.array([-np.sin(theta),np.cos(theta)])
        x,y=plane['basis']@a,plane['basis']@b
        # Resolve the inherent 180-degree symmetry using the camera-side convention.
        if x@plane['basis'][:,0]<0:x,y=-x,-y
        T=np.eye(4)
        T[:3,:3]=np.column_stack((x,y,plane['n']))
        T[:3,3]=plane['origin']+plane['basis']@center
        return T

    def _physical_edges(self,T,obs,refine=False):
        h,w=obs['mask'].shape
        supports=[];errors=[];pairs=[];visibility=[];spans=[]
        for edge_id,points in enumerate(self._edge_samples):
            xyz=points@T[:3,:3].T+T[:3,3]
            uv=self.camera.project(xyz)
            center3=T[:3,3]
            inward=center3-xyz
            inward/=np.maximum(np.linalg.norm(inward,axis=1,keepdims=True),1e-9)
            # A model edge needs blue on its inside and non-blue on its outside.
            # This rejects the net, centre stripe and most interior object edges.
            inside=self.camera.project(xyz+.08*inward)
            outside=self.camera.project(xyz-.055*inward)
            ok=(xyz[:,2]>.2)&np.all(np.isfinite(uv),axis=1)
            for arr in (uv,inside,outside):
                ok&=(arr[:,0]>6)&(arr[:,0]<w-7)&(arr[:,1]>6)&(arr[:,1]<h-7)
            visibility.append(float(np.mean(ok)))
            inds=np.flatnonzero(ok)
            if len(inds)<8:
                supports.append(0.);spans.append(0.);continue
            ui=np.rint(uv[inds]).astype(int)
            vi=np.rint(inside[inds]).astype(int);vo=np.rint(outside[inds]).astype(int)
            transition=(obs['mask'][vi[:,1],vi[:,0]]>0)&(obs['mask'][vo[:,1],vo[:,0]]==0)
            dist=obs['distance'][ui[:,1],ui[:,0]]
            good=transition&(dist<=self.edge_tolerance)
            supports.append(float(good.sum()/len(inds)))
            good_indices=inds[good]
            span=(good_indices.max()-good_indices.min())/64 if len(good_indices)>1 else 0
            spans.append(float(span))
            # Odd samples are held out from the pose refinement.
            errors.extend(dist[good & (inds%2==1)].tolist())
            if refine:
                for j in good_indices[good_indices%2==0]:
                    u,v=np.rint(uv[j]).astype(int)
                    cut=obs['edge'][v-6:v+7,u-6:u+7]
                    ys,xs=np.nonzero(cut)
                    if not len(xs):continue
                    k=np.argmin((xs-6)**2+(ys-6)**2)
                    pairs.append((edge_id,np.array([u+xs[k]-6,v+ys[k]-6],float)))
        return supports,errors,pairs,visibility,spans

    def validate(self,T,obs,initial=False):
        xyz=self.corners@T[:3,:3].T+T[:3,3]
        if not np.all(np.isfinite(T)) or np.any(xyz[:,2]<.2):return None,"behind_camera"
        plane=obs['plane']
        normal_error=angle_between(T[:3,2],plane['n'])
        center_error=abs(float(T[:3,3]@plane['n']+plane['d']))
        # Near-only depth may bias an extrapolated far plane. Validate a visual
        # rectangle using independent measured points, not only plane offset.
        observed=plane['held']
        observed=observed[np.abs(observed@plane['n']+plane['d'])<.025]
        pose_residual=np.abs((observed-T[:3,3])@T[:3,2])
        depth_error=float(np.median(pose_residual)) if len(pose_residual) else float('inf')
        if normal_error>np.deg2rad(5) or center_error>.065 or depth_error>.025:
            return None,"plane_pose_disagreement"
        local=(plane['points']-T[:3,3])@T[:3,:3]
        contained=(np.abs(local[:,0])<self.length/2+.05)&(np.abs(local[:,1])<self.width/2+.05)
        containment=float(np.mean(contained))
        if containment<.92:return None,"table_points_outside_rectangle"
        supports,errors,_,visible,spans=self._physical_edges(T,obs)
        sides=[i for i in range(4) if supports[i]>=.42 and spans[i]>=.20 and visible[i]>=.20]
        if initial:
            if len(sides)<3:return None,"need_three_supported_outer_edges"
        elif not (any(i in sides for i in (0,2)) and any(i in sides for i in (1,3))):
            return None,"insufficient_nonparallel_edges"
        if not errors:return None,"no_held_out_boundary_samples"
        error=float(np.median(errors))
        if error>self.edge_tolerance:return None,"boundary_reprojection_failed"
        uv=self.camera.project(xyz)
        h,w=obs['mask'].shape
        inferred=[not(6<p[0]<w-7 and 6<p[1]<h-7 and obs['distance'][int(round(p[1])),int(round(p[0]))]<=self.edge_tolerance)
                  for p in uv]
        return dict(edge_support=supports,visible_edges=sides,edge_visibility=visible,
                    edge_error_px=error,plane_validation_error_m=depth_error,
                    plane_inlier_ratio=plane['plane_inlier_ratio'],containment=containment,
                    normal_error_deg=float(np.rad2deg(normal_error)),center_plane_error_m=center_error,
                    corner_inferred=inferred,observed_edge_count=len(sides)),None

    def _refine(self,T,obs):
        plane=obs['plane'];basis=plane['basis']
        theta=math.atan2(T[:3,0]@basis[:,1],T[:3,0]@basis[:,0])
        c=(T[:3,3]-plane['origin'])@basis
        for _ in range(4):
            _,_,pairs,_,_=self._physical_edges(T,obs,refine=True)
            if len(pairs)<16:break
            ids=np.array([p[0] for p in pairs])
            xyz=self.camera.intersect([p[1] for p in pairs],plane['n'],plane['d'])
            xy=(xyz-plane['origin'])@basis
            a=np.array([np.cos(theta),np.sin(theta)]);b=np.array([-np.sin(theta),np.cos(theta)])
            x=(xy-c)@a;y=(xy-c)@b
            rows=[];residual=[]
            for i,xx,yy in zip(ids,x,y):
                if i in (0,2):
                    rows.append([-b[0],-b[1],-xx]);residual.append(yy-(-self.width/2 if i==0 else self.width/2))
                else:
                    rows.append([-a[0],-a[1],yy]);residual.append(xx-(self.length/2 if i==1 else -self.length/2))
            J=np.array(rows);r=np.array(residual)
            weights=np.minimum(1.,.02/np.maximum(np.abs(r),1e-8))
            delta=np.linalg.lstsq(J*weights[:,None],-r*weights,rcond=None)[0]
            c+=np.clip(delta[:2],-.025,.025);theta+=float(np.clip(delta[2],-.02,.02))
            T=self._pose(theta,c,plane)
        return T

    def _quad_refine(self,prior,obs):
        """Fit actual four supporting lines when all four are visible.

        Line intersections can be outside the image, so a clipped corner does
        not prevent initialization. PnP proposals still pass held-out depth and
        boundary checks. There is no unconstrained iterative tracking here.
        """
        uv=self.camera.project(self.corners@prior[:3,:3].T+prior[:3,3])
        choices=[]
        for i in range(4):
            a,b=uv[i],uv[(i+1)%4]
            direction=b-a;direction/=np.linalg.norm(direction)
            normal=np.array([-direction[1],direction[0]])
            group=[]
            for j,line in enumerate(obs['pixel_lines']):
                p,q=line.reshape(2,2);d=q-p;length=np.linalg.norm(d)
                if length<50 or abs(direction@(d/length))<np.cos(np.deg2rad(12)):continue
                distance=abs(((p+q)/2-a)@normal)
                if distance>45:continue
                coeff=np.cross(np.r_[p,1.],np.r_[q,1.]);coeff/=np.linalg.norm(coeff[:2])
                group.append((distance-.015*length,j,coeff))
            group.sort(key=lambda c:c[0]);unique=[]
            for entry in group:
                if all(abs(abs(entry[2]@np.r_[(uv[i]+uv[(i+1)%4])/2,1.])-abs(other[2]@np.r_[(uv[i]+uv[(i+1)%4])/2,1.]))>1.5 for other in unique):
                    unique.append(entry)
                if len(unique)>=4:break
            if not unique:return []
            choices.append(unique)
        result=[]
        for selection in itertools.product(*choices):
            if len(set(p[1] for p in selection))<4:continue
            quad=[]
            for i in range(4):
                p=np.cross(selection[(i-1)%4][2],selection[i][2])
                if abs(p[2])<1e-5:break
                quad.append(p[:2]/p[2])
            if len(quad)!=4:continue
            quad=np.array(quad,float)
            if np.max(np.abs(quad))>10000 or not cv2.isContourConvex(quad.astype(np.float32)):continue
            try:
                count,rv,tv,_=cv2.solvePnPGeneric(self.corners,quad,self.camera.K,self.camera.dist,flags=cv2.SOLVEPNP_IPPE)
            except cv2.error:continue
            for r,t in zip(rv,tv):
                T=np.eye(4);T[:3,:3]=cv2.Rodrigues(r)[0];T[:3,3]=t.reshape(3)
                metrics,_=self.validate(T,obs,True)
                if metrics:
                    score=sum(metrics['edge_support'])-.12*metrics['edge_error_px']-5*metrics['plane_validation_error_m']
                    metrics['initialization_method']='four_outer_lines_depth_checked'
                    result.append((score,T,metrics))
        return result

    def detect(self,obs):
        lines=obs['lines'];cloud=obs['cloud'];plane=obs['plane']
        if len(lines)<3:return None,None,"not_enough_boundary_lines"
        angles=[]
        for _,direction,_ in lines:
            t=math.atan2(direction[1],direction[0])
            for theta in (t,t+np.pi/2):
                theta=(theta+np.pi/2)%np.pi-np.pi/2
                if all(abs(np.sin(theta-old))>np.sin(np.deg2rad(1.5)) for old in angles):angles.append(theta)
        proposals=[];rough=[]
        for theta in angles[:24]:
            a=np.array([np.cos(theta),np.sin(theta)]);b=np.array([-np.sin(theta),np.cos(theta)])
            qx,qy=cloud@a,cloud@b
            # Keep >=97% of the observed plane cloud inside the rectangle.
            xlo,xhi=np.percentile(qx,[2,98]);ylo,yhi=np.percentile(qy,[2,98])
            if xhi-xlo>self.length+.10 or yhi-ylo>self.width+.10:continue
            xs=[];ys=[]
            for mean,direction,length in lines:
                if abs(direction@b)>np.cos(np.deg2rad(6)):xs.extend([mean@a-self.length/2,mean@a+self.length/2])
                if abs(direction@a)>np.cos(np.deg2rad(6)):ys.extend([mean@b-self.width/2,mean@b+self.width/2])
            xs=sorted(set(round(float(v),3) for v in xs if xhi-self.length/2-.04<=v<=xlo+self.length/2+.04))
            ys=sorted(set(round(float(v),3) for v in ys if yhi-self.width/2-.04<=v<=ylo+self.width/2+.04))
            for x in xs:
                for y in ys:
                    T=self._pose(theta,x*a+y*b,plane)
                    metrics,reason=self.validate(T,obs,initial=True)
                    if metrics:
                        score=sum(metrics['edge_support'])-.12*metrics['edge_error_px']
                        proposals.append((score,T,metrics))
                    else:
                        weak,_=self.validate(T,obs,initial=False)
                        if weak:rough.append((sum(weak['edge_support'])-.12*weak['edge_error_px'],T,weak))
        if not proposals and not rough:return None,None,"no_metric_rectangle_with_three_outer_edges"
        proposals.sort(key=lambda c:-c[0]);self.last_candidates=proposals[:5]
        refined=[]
        rough.extend(proposals);rough.sort(key=lambda c:-c[0])
        for _,T,_ in rough[:4]:refined.extend(self._quad_refine(T,obs))
        for _,T,_ in proposals[:5]:
            original=T.copy()
            T=self._refine(T,obs)
            metrics,_=self.validate(T,obs,initial=True)
            if metrics is None:
                T=original;metrics,_=self.validate(T,obs,initial=True)
            if metrics:
                metrics['initialization_method']='three_or_four_metric_outer_edges'
                refined.append((sum(metrics['edge_support'])-.12*metrics['edge_error_px'],T,metrics))
        if not refined:return None,None,"refinement_failed_independent_validation"
        refined.sort(key=lambda c:-c[0])
        score,T,metrics=refined[0]
        for score2,T2,_ in refined[1:]:
            distinct=np.linalg.norm(T[:3,3]-T2[:3,3])>.12 or rotation_distance(T[:3,:3],T2[:3,:3])>np.deg2rad(8)
            if distinct and score-score2<.20:return None,None,"ambiguous_edge_identity"
        return T,metrics,None

    def _metadata(self,timestamp_s):
        stale=None if self._last_verified is None else max(0.,timestamp_s-self._last_verified)
        state=self._state
        if state=='VALID' and stale is not None and stale>self.hold_seconds:state='LOST'
        valid=state=='VALID'
        return dict(self._metrics,state=state,valid=valid,table_frame_id=self._epoch,
                    measured_this_frame=valid and stale is not None and stale<.06,
                    stale_s=stale,reject_reason=self._reason,motion_mode='fixed',
                    verification_timestamp_s=self._last_verified,
                    candidate_confirmations=len(self._pending),
                    sdk_projection_error_px=self.camera.sdk_error_px)

    def snapshot(self,timestamp_s):
        with self._lock:
            meta=self._metadata(timestamp_s)
            if self._T is None or not meta['valid']:return None,meta
            confidence=float(np.clip(.4+.12*len(meta.get('visible_edges',[]))-.035*meta.get('edge_error_px',0),0,.95))
            snap=TableSnapshot(self._T.copy(),timestamp_s,np.zeros(3),np.zeros(3),confidence,
                               self._last_verified,dict(meta))
            return snap,meta

    def predict(self,timestamp_s):return self.snapshot(timestamp_s)[0]

    def request_reinitialize(self):
        with self._lock:self._force=True

    def load_initial_pose(self,timestamp_s):
        # An old pose is a candidate, never accepted merely because a file exists.
        try:
            with np.load(self.pose_file) as data:
                T=np.asarray(data['T_camera_table'],float)
                for key,expected in (('table_length',self.length),('table_width',self.width)):
                    if key in data and abs(float(data[key])-expected)>1e-5:return False
            if T.shape!=(4,4) or not np.all(np.isfinite(T)):return False
            with self._lock:
                self._loaded_candidate=T.copy();self._T=None;self._pending=[]
                self._state='SEARCHING';self._reason='saved_pose_requires_revalidation'
            return True
        except (OSError,KeyError,ValueError):return False

    def save_pose(self):
        with self._lock:
            if self._T is None or self._state!='VALID':return False
            self.pose_file.parent.mkdir(parents=True,exist_ok=True)
            np.savez(self.pose_file,T_camera_table=self._T,table_length=self.length,
                     table_width=self.width,table_frame_id=self._epoch)
            return True

    def update(self,bgr,depth,timestamp_s):
        with self._lock:
            if self._force:
                self._T=None;self._pending=[];self._loaded_candidate=None;self._state='SEARCHING';self._force=False
            if self._last_attempt is not None and timestamp_s-self._last_attempt<1/self.validation_hz:return None
            self._last_attempt=timestamp_s
            T=None if self._T is None else self._T.copy()
            old_state=self._state
        obs,reason=self.observe(bgr,depth)
        self.last_observation=obs
        candidate=None;metrics=None
        if obs is not None and T is not None:
            metrics,reason=self.validate(T,obs,initial=False)
        if metrics is not None:
            with self._lock:
                self._state='VALID';self._last_verified=timestamp_s;self._reason=None;self._metrics=metrics;self._pending=[]
        else:
            # Geometry failure never updates the locked pose. Full search is
            # independent of its centre and orientation.
            detect_reason=reason
            if obs is not None:
                if self._loaded_candidate is not None:
                    metrics,detect_reason=self.validate(self._loaded_candidate,obs,initial=True)
                    if metrics is not None:candidate=self._loaded_candidate.copy()
                    else:self._loaded_candidate=None
                if candidate is None:candidate,metrics,detect_reason=self.detect(obs)
            with self._lock:
                self._reason=reason or detect_reason
                self._state='HOLDING' if self._T is not None and self._last_verified is not None and timestamp_s-self._last_verified<=self.hold_seconds else ('LOST' if self._T is not None else 'SEARCHING')
                if candidate is None:self._pending=[]
                else:
                    if self._pending and (np.linalg.norm(candidate[:3,3]-self._pending[-1][:3,3])>.05 or rotation_distance(candidate[:3,:3],self._pending[-1][:3,:3])>np.deg2rad(3)):
                        self._pending=[]
                    self._pending.append(candidate)
                    if self._T is None:self._state='CANDIDATE'
                    if len(self._pending)>=self.confirm_frames:
                        # Use the medoid (a validated rigid pose), then verify it
                        # against the current frame before locking.
                        centers=np.array([p[:3,3] for p in self._pending])
                        idx=int(np.argmin(np.linalg.norm(centers-np.median(centers,axis=0),axis=1)))
                        locked=self._pending[idx]
                        accepted,_=self.validate(locked,obs,initial=True)
                        if accepted is not None:
                            self._T=locked.copy();self._epoch+=1;self._state='VALID';self._last_verified=timestamp_s
                            self._reason=None;self._metrics=accepted;self._pending=[];self._loaded_candidate=None
            if self._reason is None and metrics is None:self._reason=detect_reason
        with self._lock:
            meta=self._metadata(timestamp_s)
            self._debug=self.draw(bgr,obs,candidate,meta)
            if self._state!=old_state:
                print(f"[TABLE] {old_state} -> {self._state} epoch={self._epoch} reason={self._reason}")
        return meta

    def draw(self,bgr,obs,candidate,meta):
        image=bgr.copy()
        if obs is not None:
            observed_edges=obs['edge'].copy()
            if self._T is not None:
                projected=self.camera.project(self.corners@self._T[:3,:3].T+self._T[:3,3])
                if np.all(np.isfinite(projected)) and np.max(np.abs(projected))<100000:
                    band=np.zeros(image.shape[:2],np.uint8)
                    cv2.polylines(band,[np.rint(projected).astype(np.int32)],True,255,30)
                    observed_edges &= band
            image[observed_edges>0]=(180,180,0)
        poses=[]
        if candidate is not None and not meta['valid']:poses.append((candidate,(0,0,220)))
        if self._T is not None:poses.append((self._T,(0,255,0) if meta['valid'] else (140,140,140)))
        h,w=image.shape[:2]
        for T,color in poses:
            xyz=self.corners@T[:3,:3].T+T[:3,3]
            if np.any(xyz[:,2]<.2):continue
            uv=self.camera.project(xyz)
            if not np.all(np.isfinite(uv)) or np.max(np.abs(uv))>1e6:continue
            for i in range(4):
                a=tuple(int(v) for v in np.rint(uv[i]));b=tuple(int(v) for v in np.rint(uv[(i+1)%4]))
                visible,a,b=cv2.clipLine((0,0,w,h),a,b)
                if visible:cv2.line(image,a,b,color,2)
                p=tuple(int(v) for v in np.rint(uv[i]))
                if 0<=p[0]<w and 0<=p[1]<h:
                    inferred=meta.get('corner_inferred',[True]*4)[i]
                    cv2.circle(image,p,5,color,2 if inferred else -1)
                    cv2.putText(image,f"{i}{'*' if inferred else ''}",(p[0]+6,p[1]-6),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
            center=self.camera.project(T[:3,3][None])[0]
            if np.all(np.isfinite(center)) and np.max(np.abs(center))<1e6:
                cv2.drawMarker(image,tuple(int(v) for v in np.rint(center)),color,cv2.MARKER_CROSS,18,2)
        lines=[f"Table {meta['state']} | fixed | epoch {meta['table_frame_id']}",
               f"edges {meta.get('visible_edges',[])} error {meta.get('edge_error_px',float('nan')):.2f}px depth {1000*meta.get('plane_validation_error_m',float('nan')):.1f}mm",
               str(meta.get('reject_reason') or 'independent boundary/depth verification passed')]
        for j,text in enumerate(lines):
            cv2.putText(image,text,(12,26+24*j),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,0,0),3)
            cv2.putText(image,text,(12,26+24*j),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,255,255),1)
        return image

    def get_debug_image(self):
        with self._lock:return None if self._debug is None else self._debug.copy()
