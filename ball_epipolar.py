"""Experimental RGB-guided local stereo: observed patches, no diameter-derived Z."""
import cv2
import time
import numpy as np
import pyrealsense2 as rs


def project(intr,p):
    return np.array(rs.rs2_project_point_to_pixel(intr,list(map(float,p))))


def deproject(intr,uv,z):
    return np.array(rs.rs2_deproject_pixel_to_point(intr,list(map(float,uv)),float(z)))


def transform(T,p):return T[:3,:3]@p+T[:3,3]


class EpipolarBallMatcher:
    def __init__(self,model):
        self.m=model;self.diagnostics={}
        # D455 IR intrinsics are undistorted in this recording; exact geometry
        # and projection checks are still applied to every candidate pair.
        self.K1=self.K(model.ir_left_intr);self.K2=self.K(model.ir_right_intr)
        self.P1=self.K1@np.c_[np.eye(3),np.zeros(3)]
        self.P2=self.K2@model.T_left_right[:3,:]

    @staticmethod
    def K(i):return np.array([[i.fx,0,i.ppx],[0,i.fy,i.ppy],[0,0,1.]])

    def right_hint(self,uv):
        """Optional independently estimated right location; None searches the epipolar interval."""
        return None

    def roi(self,image,bbox,bounds=(.3,6.)):
        x,y,w,h=bbox;points=[]
        for uv in ((x,y),(x+w,y),(x,y+h),(x+w,y+h)):
            for z in bounds:
                p=transform(self.m.T_left_color,deproject(self.m.color_intr,uv,z))
                points.append(project(self.m.ir_left_intr,p))
        points=np.array(points)
        a=np.maximum(0,np.floor(points.min(0)-6)).astype(int)
        b=np.minimum(image.shape[::-1],np.ceil(points.max(0)+7)).astype(int)
        return a,b

    def proposals(self,image,bbox,bounds=(.3,6.)):
        a,b=self.roi(image,bbox,bounds);x0,y0=a;x1,y1=b
        if x1-x0<7 or y1-y0<7:return []
        crop=image[y0:y1,x0:x1].astype(np.float32)
        smooth=cv2.GaussianBlur(crop,(0,0),.7)
        signal=smooth-cv2.GaussianBlur(smooth,(0,0),5.)
        maxima=(signal>=cv2.dilate(signal,np.ones((5,5),np.uint8))-1e-5)&(signal>.65)
        yy,xx=np.where(maxima);order=np.argsort(signal[yy,xx])[::-1]
        points=[]
        for n in order[:60]:
            x,y=int(xx[n]),int(yy[n]);peak=float(signal[y,x])
            if x<2 or y<2 or x>=crop.shape[1]-2 or y>=crop.shape[0]-2:continue
            sx0,sy0=max(0,x-8),max(0,y-8);patch=signal[sy0:y+9,sx0:x+9]
            weights=np.maximum(patch-.35*peak,0)
            ry,rx=np.indices(weights.shape);den=weights.sum()
            uv=np.array([sx0+(weights*rx).sum()/den+x0,sy0+(weights*ry).sum()/den+y0])
            if all(np.linalg.norm(uv-p['uv'])>3 for p in points):points.append(dict(uv=uv,contrast=peak))
            if len(points)>=24:break
        return points

    def match(self,left,right,rgb_result,candidate,ir_t,rgb_t,bounds=(.3,6.),deadline=None):
        self.diagnostics=dict(reason='no_image_detection',left_proposals=0,pairs=0)
        if not rgb_result.get('valid') or candidate is None:return None
        if abs(ir_t-rgb_t)>.018:
            self.diagnostics['reason']='timestamp_mismatch';return None
        points=self.proposals(left,candidate['bbox'],bounds);self.diagnostics['left_proposals']=len(points)
        accepted=[];rejected={}
        def reject(reason):rejected[reason]=rejected.get(reason,0)+1
        for c in points:
            if deadline is not None and time.perf_counter()>=deadline:
                self.diagnostics['reason']='deadline_exhausted';return None
            uv=c['uv']
            # Use image width only to choose a patch radius. It never yields Z.
            radius=int(np.clip(candidate['minor_px']*self.m.ir_left_intr.fx/self.m.color_intr.fx*.7,6,12))
            if np.any(uv<radius) or np.any(uv>=np.array(left.shape[::-1])-radius):continue
            hint=self.right_hint(uv)
            if hint is None:
                ray=deproject(self.m.ir_left_intr,uv,1.)
                ex=np.array([project(self.m.ir_right_intr,transform(self.m.T_left_right,ray*z)) for z in bounds])
            else:ex=np.array([hint,hint])
            if not np.isfinite(ex).all():reject('invalid_stereo_hint');continue
            x0=max(0,int(np.floor(ex[:,0].min()))-2-radius)
            x1=min(right.shape[1],int(np.ceil(ex[:,0].max()))+3+radius)
            y0=max(0,int(np.floor(ex[:,1].min()))-2-radius);y1=min(right.shape[0],int(np.ceil(ex[:,1].max()))+3+radius)
            if min(x1-x0,y1-y0)<2*radius+1:continue
            template=cv2.getRectSubPix(left,(2*radius+1,2*radius+1),tuple(map(float,uv)))
            if template.std()<.5:reject('weak_patch');continue
            scores=cv2.matchTemplate(right[y0:y1,x0:x1],template,cv2.TM_CCOEFF_NORMED)
            maxima=(scores>=cv2.dilate(scores,np.ones((3,3),np.uint8))-1e-6)&(scores>.65)
            ys,xs=np.where(maxima);indices=np.argsort(scores[ys,xs])[::-1][:5]
            for n in indices:
                ix,iy=int(xs[n]),int(ys[n]);ncc=float(scores[iy,ix]);shift=np.zeros(2)
                for ax in (0,1):
                    if ax==0 and 0<ix<scores.shape[1]-1:a,b,d=scores[iy,ix-1:ix+2]
                    elif ax==1 and 0<iy<scores.shape[0]-1:a,b,d=scores[iy-1:iy+2,ix]
                    else:continue
                    denom=a-2*b+d
                    if abs(denom)>1e-5:shift[ax]=np.clip(.5*(a-d)/denom,-.5,.5)
                vr=np.array([x0+ix+radius,y0+iy+radius])+shift
                q=cv2.triangulatePoints(self.P1,self.P2,uv.reshape(2,1),vr.reshape(2,1))[:,0]
                if abs(q[3])<1e-8:continue
                p=q[:3]/q[3]
                if not .3<p[2]<6:reject('range');continue
                pr=transform(self.m.T_left_right,p)
                error=max(np.linalg.norm(project(self.m.ir_left_intr,p)-uv),np.linalg.norm(project(self.m.ir_right_intr,pr)-vr))
                if error>1.2:reject('epipolar');continue
                pc=transform(self.m.T_color_left,p);color_uv=project(self.m.color_intr,pc)
                dist=float(np.linalg.norm(color_uv-rgb_result['uv']))
                # RGB blur describes a region occupied over exposure; bound
                # association by its observed extent, never a whole-table ROI.
                allowed=max(5.,.55*candidate['major_px'])
                if dist>allowed:reject('rgb_position');continue
                expected=self.m.color_intr.fx*.04/pc[2]
                size_error=abs(candidate['minor_px']-expected)/(2+.4*expected)
                if size_error>1:reject('physical_size');continue
                width=self.m.ir_left_intr.fx*.04/p[2]
                object_radius=int(np.clip(width*1.25,8,32))
                patches=[cv2.getRectSubPix(im,(2*object_radius+1,2*object_radius+1),tuple(map(float,x))).astype(float)
                    for im,x in ((left,uv),(right,vr))]
                rr=np.sum(np.array(np.mgrid[-object_radius:object_radius+1,-object_radius:object_radius+1])**2,axis=0)
                core=rr<=max(1.,.25*width)**2
                ring=(rr>=max(2.,.65*width)**2)&(rr<=object_radius**2)
                if not ring.any():reject('patch_too_small');continue
                contrasts=[float(np.median(v[core])-np.median(v[ring])) for v in patches]
                noise=[max(.5,1.4826*np.median(abs(v[ring]-np.median(v[ring])))) for v in patches]
                limits=[max(.75,3.5*n/np.sqrt(core.sum())) for n in noise]
                if any(c<limit for c,limit in zip(contrasts,limits)):reject('no_positive_object');continue
                if min(contrasts)<3 and ncc<.8:reject('weak_object_match');continue
                shapes=[]
                for patch in patches:
                    background=float(np.median(patch[ring]));peak=float(np.max(patch[core]))
                    binary=np.uint8(patch>background+max(.5,.35*(peak-background)))
                    _,components,stats,_=cv2.connectedComponentsWithStats(binary,8)
                    label=components[object_radius,object_radius]
                    if label==0:shapes.append(False);continue
                    x,y,bw,bh,area=stats[label]
                    contour=cv2.findContours(np.uint8(components==label),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0][0]
                    minor,major=sorted(np.array(cv2.minAreaRect(contour)[1])+1)
                    shapes.append(bool(x>0 and y>0 and x+bw<patch.shape[1] and y+bh<patch.shape[0]
                        and area>=6 and minor>=3 and major/minor<=3.5
                        and abs(minor-width)<=2.+.5*width))
                if not all(shapes):reject('ir_object_shape_or_size');continue
                cost=(1-ncc)+.25*dist/allowed+.15*size_error+.15*error
                accepted.append(dict(cost=float(cost),ncc=ncc,xyz_ir=p.tolist(),left_uv=uv.tolist(),right_uv=vr.tolist(),
                    rgb_projection=color_uv.tolist(),rgb_error_px=dist,size_error=float(size_error),contrast=contrasts,
                    ir_t=ir_t,reprojection_px=float(error)))
        accepted.sort(key=lambda x:x['cost']);self.diagnostics.update(pairs=len(accepted),rejected=rejected,reason='no_stereo_match')
        if not accepted:return None
        best=accepted[0];alternatives=[a for a in accepted[1:] if abs(a['xyz_ir'][2]-best['xyz_ir'][2])>.12]
        if alternatives and alternatives[0]['cost']-best['cost']<.08:
            self.diagnostics.update(reason='ambiguous_depth',best=best,alternative=alternatives[0]);return None
        self.diagnostics.update(reason='verified_stereo',best=best)
        return best
