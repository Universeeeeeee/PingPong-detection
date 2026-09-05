"""RGB-only ball localization. Pixel detections never imply a valid depth.

Reconstruct the whole colour component before measuring shape. Orange blurred
over a blue background can cross the hue wrap into pink; narrow HSV islands are
not the object's boundary. No scene coordinates or prerecorded paths are used.
"""
from collections import deque
import math
import cv2
import numpy as np


class ImageBallDetector:
    def __init__(self, hsv_ranges, orange=False, retain_contours=False):
        self.ranges=hsv_ranges;self.orange=orange
        self.retain_contours=retain_contours;self.selected_candidate=None
        self.previous=None;self.history=deque(maxlen=4);self.track_id=0
        self.last_timestamp=None;self.background=None;self.last_strong=None

    def detect(self,bgr,timestamp_s):
        if self.last_timestamp is not None and timestamp_s<=self.last_timestamp:
            return self.empty(timestamp_s,'nonmonotonic_frame'),[],np.zeros(bgr.shape[:2],np.uint8)
        if self.previous is not None and self.previous.shape!=bgr.shape[:2]:
            self.previous=None;self.background=None;self.history.clear();self.last_strong=None
        h,w=bgr.shape[:2];hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
        self.selected_candidate=None
        strict=np.zeros((h,w),np.uint8)
        for lo,hi in self.ranges:strict|=cv2.inRange(hsv,lo,hi)
        mask=strict.copy()
        if self.orange:
            mask=cv2.inRange(hsv,np.array([0,35,55],np.uint8),np.array([40,255,255],np.uint8))
            mask|=cv2.inRange(hsv,np.array([150,35,55],np.uint8),np.array([179,255,255],np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        motion=np.zeros_like(gray) if self.previous is None else cv2.absdiff(gray,self.previous)
        self.previous=gray
        masks=[('whole',mask)];novel=np.zeros((h,w),np.uint8)
        if self.orange:
            colour=bgr.astype(np.float32)
            chromatic=colour[:,:,2]-colour[:,:,1]
            for level in (45,60,75):
                cm=np.where((mask>0)&(chromatic>=level),255,0).astype(np.uint8)
                if level!=60:cm=cv2.morphologyEx(cm,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
                masks.append(('chroma',cm))
            if self.background is not None:
                delta=colour-self.background
                foreground=(delta[:,:,2]-delta[:,:,1]>18)&(delta[:,:,2]-delta[:,:,0]>18)
                foreground&=(mask>0)&(chromatic>20)
                novel=foreground.astype(np.uint8)
                fm=np.where(foreground,255,0).astype(np.uint8)
                # Small achromatic gaps (white table lines/highlights) may split
                # one moving streak. Reconstruction still passes shape/context.
                fm=cv2.morphologyEx(fm,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
                masks.append(('foreground',fm))
                cv2.accumulateWeighted(colour,self.background,.015)
            else:self.background=colour.copy()
        entries=[]
        for kind,segmentation in masks:
            contours,_=cv2.findContours(segmentation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            entries.extend((kind,segmentation,c) for c in contours)
        candidates=[]
        for kind,segmentation,contour in entries:
            area=cv2.contourArea(contour)
            if not 8<=area<=5000:continue
            x,y,bw,bh=cv2.boundingRect(contour)
            rect=cv2.minAreaRect(contour);minor,major=sorted(np.array(rect[1])+1.)
            if not 4<=minor<=100 or major/minor>9:continue
            hull=cv2.contourArea(cv2.convexHull(contour));solidity=area/max(hull,1.)
            fill=area/(minor*major)
            if solidity<.80 or fill<.48:continue
            pad=max(5,int(minor*.6));x0=max(0,x-pad);y0=max(0,y-pad);x1=min(w,x+bw+pad);y1=min(h,y+bh+pad)
            inside=np.zeros((y1-y0,x1-x0),np.uint8)
            local=contour-np.array([[[x0,y0]]]);cv2.drawContours(inside,[local],-1,255,-1)
            region=inside>0
            seed=float(np.mean(strict[y0:y1,x0:x1][region]>0))
            moving=float(np.mean(motion[y0:y1,x0:x1][region]>12))
            new_colour=float(np.mean(novel[y0:y1,x0:x1][region]>0))
            pixels=bgr[y0:y1,x0:x1][region].astype(float)
            chroma=float(np.percentile(pixels[:,2]-pixels[:,1],75))
            # Red/pink-only streaks require observed motion and later association.
            if seed<.03 and (not self.orange or moving<.25 or chroma<25):continue
            distance=cv2.distanceTransform(255-inside,cv2.DIST_L2,3)
            ring=(distance>=2)&(distance<=max(4,minor*.5))
            context=float(np.mean(segmentation[y0:y1,x0:x1][ring]>0)) if ring.any() else 1.
            perimeter=cv2.arcLength(contour,True);circularity=4*math.pi*area/max(perimeter**2,1.)
            moment=cv2.moments(contour)
            uv=np.array([moment['m10']/moment['m00'],moment['m01']/moment['m00']])
            aspect=major/minor;clipped=x<=1 or y<=1 or x+bw>=w-1 or y+bh>=h-1
            compact=aspect<=1.7
            reason=None
            if context>.30:reason='surrounding_colour'
            elif compact and circularity<.50:reason='irregular_boundary'
            elif not compact and moving<.12:reason='stationary_streak'
            elif self.orange and chroma<30:reason='weak_orange_chroma'
            quality=.30*solidity+.25*min(1.,fill/.72)+.25*min(1.,chroma/80.)+.20*moving
            strong=reason is None and not clipped and seed>=.20
            if self.orange:
                static=(kind=='whole' and aspect<=1.45 and circularity>=.70 and chroma>=65)
                dynamic=(seed>=.30 and chroma>=50 and new_colour>=.35 and moving>=.25)
                strong=strong and (static or (dynamic and quality>=.80))
            candidates.append(dict(uv=uv.tolist(),bbox=[x,y,bw,bh],minor_px=float(minor),
                major_px=float(major),aspect=float(aspect),solidity=float(solidity),
                circularity=float(circularity),context=context,chroma=chroma,seed_fraction=seed,
                motion=moving,new_colour=new_colour,quality=float(quality),strong=bool(strong),reason=reason,clipped=clipped,segmentation=kind))
            if self.retain_contours:candidates[-1]['contour']=contour
        candidates.sort(key=lambda c:c['quality'],reverse=True)
        result=self.select(candidates,timestamp_s)
        return result,candidates,mask

    def select(self,candidates,t):
        # Sensor timestamps identify independent images; no repeated confidence.
        if self.last_timestamp is not None and t<=self.last_timestamp:
            return self.empty(t,'nonmonotonic_frame')
        self.last_timestamp=t
        if self.history and t-self.history[-1][0]>.20:self.history.clear()
        ranked=[]
        for c in candidates:
            if c['reason'] is not None:continue
            uv=np.array(c['uv']);association=False;distance=0.
            if self.history:
                last_t,last_uv,last_width=self.history[-1];dt=t-last_t
                velocity=np.zeros(2)
                if len(self.history)>=2:
                    prev_t,prev_uv,_=self.history[-2]
                    velocity=(last_uv-prev_uv)/max(last_t-prev_t,.001)
                prediction=last_uv+dt*velocity
                distance=min(np.linalg.norm(uv-prediction),np.linalg.norm(uv-last_uv)+10.)
                allowed=max(25.,2*c['minor_px'])+np.linalg.norm(velocity)*dt+2000.*dt*dt
                association=distance<allowed and .25<c['minor_px']/last_width<2.5
            if not c['strong'] and not association:continue
            hold=.30 if c['clipped'] else .18
            if not c['strong'] and self.last_strong is not None and t-self.last_strong>hold:continue
            if self.orange and not c['strong']:
                if not association:continue
                if np.linalg.norm(velocity)>100 and c['new_colour']<.25:continue
                if c['chroma']<50 or c['seed_fraction']<.03:
                    if c['new_colour']<.65 or np.linalg.norm(velocity)<100:continue
            score=c['quality']+(0.35*math.exp(-distance/40.) if association else 0.)
            if c.get('segmentation')=='whole':score+=.015
            ranked.append((score,c,association))
        ranked.sort(key=lambda q:q[0],reverse=True)
        if not ranked:return self.empty(t,'no_image_evidence')
        score,c,associated=ranked[0]
        if c['quality']<(.65 if associated else .72):return self.empty(t,'weak_identity')
        alternatives=[q for q in ranked[1:] if np.linalg.norm(np.array(q[1]['uv'])-c['uv'])>max(12.,c['minor_px'])]
        if alternatives and score-alternatives[0][0]<.06:
            return self.empty(t,'ambiguous_image_candidates')
        if not associated:
            self.history.clear();self.track_id+=1
        width=c['minor_px'] if not associated else .7*self.history[-1][2]+.3*c['minor_px']
        self.history.append((t,np.array(c['uv']),width))
        if c['strong']:self.last_strong=t
        self.selected_candidate=c
        return dict(valid=True,measured=True,state='DETECTED',timestamp_s=t,
                    uv=c['uv'],bbox=c['bbox'],track_id=self.track_id,source='rgb_contour',
                    quality=c['quality'],reason='image_evidence',depth_valid=False)

    def empty(self,t,reason):
        return dict(valid=False,measured=False,state='LOST' if self.history else 'SEARCHING',
                    timestamp_s=t,uv=None,bbox=None,track_id=self.track_id,
                    source='none',quality=0.,reason=reason,depth_valid=False)
