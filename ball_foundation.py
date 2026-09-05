"""FFS disparity guides a local observed-patch verification, with the same ball gates."""
from collections import OrderedDict
import time
import numpy as np
from ball_epipolar import EpipolarBallMatcher,project,deproject,transform
from ffs_runtime import FFSRuntime

class FoundationBallMatcher(EpipolarBallMatcher):
    def __init__(self,model,runtime=None,width=384,height=256):
        super().__init__(model)
        if not (np.allclose(self.K1,self.K2,atol=1e-3) and
                np.allclose(model.T_left_right[:3,:3],np.eye(3),atol=1e-5) and
                np.max(np.abs(model.T_left_right[1:3,3]))<1e-4 and model.T_left_right[0,3]<0 and
                max(abs(x) for x in list(model.ir_left_intr.coeffs)+list(model.ir_right_intr.coeffs))<1e-6):
            raise ValueError('FFS adapter currently requires already rectified, matched IR intrinsics')
        self.runtime=runtime or FFSRuntime();self.width=width;self.height=height
        self.warmup_stats=self.runtime.warmup(width,height)
        self.minimum_ms=self.warmup_stats['p95']+2.
        self.cache=OrderedDict();self.active=None;self.target=None
        self.calls=0;self.budget_skips=0

    def crop(self,left,candidate):
        a,b=self.roi(left,candidate['bbox'],(.3,6.))
        h,w=left.shape;cw=min(w,self.width);ch=min(h,self.height)
        # Same origin in both views preserves disparity and the full-res scale.
        cx=(a[0]+b[0])/2-40;cy=(a[1]+b[1])/2
        x=int(np.clip(round((cx-cw/2)/16)*16,0,w-cw))
        y=int(np.clip(round((cy-ch/2)/16)*16,0,h-ch))
        return x,y,cw,ch

    def sample_disparity(self,uv):
        x,y,disp=self.active
        u,v=np.asarray(uv)-[x,y]
        if not (0<=u<disp.shape[1]-1 and 0<=v<disp.shape[0]-1):return float('nan')
        ix,iy=int(u),int(v);du,dv=u-ix,v-iy
        patch=disp[iy:iy+2,ix:ix+2]
        if not np.isfinite(patch).all() or np.min(patch)<=0:return float('nan')
        return float((1-dv)*((1-du)*patch[0,0]+du*patch[0,1])+dv*((1-du)*patch[1,0]+du*patch[1,1]))

    def right_hint(self,uv):
        d=self.sample_disparity(uv)
        return np.asarray(uv)-[d,0.]

    def proposals(self,image,bbox,bounds=(.3,6.)):
        points=super().proposals(image,bbox,bounds)
        ranked=[]
        for p in points:
            d=self.sample_disparity(p['uv'])
            if not np.isfinite(d) or d<=0:continue
            z=self.m.ir_left_intr.fx*(-self.m.T_left_right[0,3])/d
            if not bounds[0]<=z<=bounds[1]:continue
            xyz=deproject(self.m.ir_left_intr,p['uv'],z)
            uv=project(self.m.color_intr,transform(self.m.T_color_left,xyz))
            dist=float(np.linalg.norm(uv-self.target))
            ranked.append((dist,p))
        ranked.sort(key=lambda x:x[0])
        return [p for _,p in ranked[:12]]

    def match(self,left,right,rgb_result,candidate,ir_t,rgb_t,bounds=(.3,6.),deadline=None):
        self.diagnostics=dict(reason='no_image_detection',left_proposals=0,pairs=0,backend='ffs_guided_ncc')
        if not rgb_result.get('valid') or candidate is None:return None
        if abs(ir_t-rgb_t)>.018:self.diagnostics['reason']='timestamp_mismatch';return None
        x,y,cw,ch=self.crop(left,candidate);key=(ir_t,x,y,cw,ch);inference_ms=0.
        if key not in self.cache:
            if deadline is not None and (deadline-time.perf_counter())*1000<self.minimum_ms:
                self.budget_skips+=1;self.diagnostics['reason']='ffs_budget_insufficient';return None
            begin=time.perf_counter()
            disp=self.runtime.predict(left[y:y+ch,x:x+cw],right[y:y+ch,x:x+cw])
            inference_ms=1000*(time.perf_counter()-begin);self.calls+=1
            self.cache[key]=(x,y,disp)
            if len(self.cache)>8:self.cache.popitem(last=False)
        self.active=self.cache[key];self.target=np.asarray(rgb_result['uv'])
        if deadline is not None and time.perf_counter()>=deadline:
            self.diagnostics.update(reason='deadline_exhausted_after_ffs',inference_ms=inference_ms);return None
        result=super().match(left,right,rgb_result,candidate,ir_t,rgb_t,bounds,deadline)
        self.diagnostics.update(backend='ffs_guided_ncc',inference_ms=inference_ms,crop=[x,y,cw,ch])
        if result is not None:
            result['learned_disparity_px']=self.sample_disparity(result['left_uv'])
            result['backend']='ffs_guided_ncc'
        return result
