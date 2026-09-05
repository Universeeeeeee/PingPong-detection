"""Current-pixel ROI acceleration with transactional full-frame fallback."""
from collections import deque
import cv2
import numpy as np
from ball_image import ImageBallDetector


class FastImageBallDetector:
    def __init__(self):
        ranges=[(np.array([2,75,65],np.uint8),np.array([35,255,255],np.uint8))]
        self.engine=ImageBallDetector(ranges,True,True)
        self.last_full=None;self.mode='full';self.selected_candidate=None

    @property
    def history(self):return self.engine.history

    def detect(self,bgr,t):
        d=self.engine
        if d.last_timestamp is not None and t<=d.last_timestamp:
            return d.empty(t,'nonmonotonic_frame'),[],None
        h,w=bgr.shape[:2]
        if (not d.history or d.previous is None or d.previous.shape!=(h,w) or
            self.last_full is None or t-self.last_full>=1. or t-d.history[-1][0]>.10):
            result,candidates,mask=d.detect(bgr,t);self.last_full=t;self.mode='full'
        else:
            last_t,uv,width=d.history[-1];velocity=np.zeros(2)
            if len(d.history)>=2:
                a,b=list(d.history)[-2:];velocity=(b[1]-a[1])/max(.001,b[0]-a[0])
            predicted=uv+(t-last_t)*velocity
            lo=np.minimum(uv,predicted)-max(90.,4*width);hi=np.maximum(uv,predicted)+max(90.,4*width)
            x0,y0=np.maximum(0,np.floor(lo)).astype(int);x1,y1=np.minimum([w,h],np.ceil(hi)).astype(int)
            snapshot=d.__dict__.copy();offset=np.array([x0,y0])
            d.previous=snapshot['previous'][y0:y1,x0:x1]
            d.background=None if snapshot['background'] is None else snapshot['background'][y0:y1,x0:x1].copy()
            d.history=deque([(ht,p-offset,wd) for ht,p,wd in snapshot['history']],maxlen=4)
            result,candidates,mask=d.detect(bgr[y0:y1,x0:x1],t)
            c=d.selected_candidate
            if not result['valid'] or c['clipped']:
                d.__dict__.clear();d.__dict__.update(snapshot)
                result,candidates,mask=d.detect(bgr,t);self.last_full=t;self.mode='fallback'
            else:
                self.mode='roi'
                for candidate in candidates:
                    candidate['uv']=(np.array(candidate['uv'])+offset).tolist()
                    candidate['bbox']=(np.array(candidate['bbox'])+np.r_[offset,[0,0]]).tolist()
                    candidate['contour']=candidate['contour']+offset.reshape(1,1,2)
                result.update(uv=c['uv'],bbox=c['bbox'])
                d.history=deque([(ht,p+offset,wd) for ht,p,wd in d.history],maxlen=4)
                updated=d.background;d.background=snapshot['background']
                if d.background is not None:d.background[y0:y1,x0:x1]=updated
                d.previous=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        self.selected_candidate=d.selected_candidate if result['valid'] else None
        return result,candidates,mask
