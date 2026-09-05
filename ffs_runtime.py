"""Isolated Fast-FoundationStereo inference; includes transfers in wall timing."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import cv2
import numpy as np

BASE=Path(__file__).resolve().parent
DEFAULT_SOURCE=BASE/'third_party/Fast-FoundationStereo_eval'
DEFAULT_WEIGHTS=BASE/'third_party/ffs_weights/nvidia_hf/model_best_bp2_serialize.pth'
OFFICIAL_SHA256='7aee85948373da62b0503c2542507129a3e7cab9d97d10e6790d89512a7db214'

class FFSRuntime:
    def __init__(self,weights=DEFAULT_WEIGHTS,source=DEFAULT_SOURCE,iterations=4,max_disp=192):
        import torch
        self.torch=torch;self.iterations=iterations
        weights=Path(weights)
        digest=hashlib.sha256(weights.read_bytes()).hexdigest()
        if digest!=OFFICIAL_SHA256:
            raise ValueError('Expected the pinned NVIDIA public checkpoint checksum')
        sys.path.insert(0,str(Path(source).resolve()))
        import core.foundation_stereo
        from core.utils.utils import InputPadder
        self.InputPadder=InputPadder
        torch.set_num_threads(1)
        self.model=torch.load(str(weights),map_location='cpu',weights_only=False).cuda().eval()
        self.model.args.valid_iters=iterations;self.model.args.max_disp=max_disp
        self.model.args.mixed_precision=True
        # The NVIDIA HF bp2 checkpoint omits this later-added flag. Preserve
        # the upstream GWC builder's declared default rather than changing its
        # feature-normalization semantics.
        if 'normalize' not in self.model.args:
            import inspect
            from core.submodule import build_gwc_volume_optimized_pytorch1
            self.model.args.normalize=inspect.signature(build_gwc_volume_optimized_pytorch1).parameters['normalize'].default
        self.sha256=digest

    def predict(self,left,right):
        torch=self.torch
        if left.shape!=right.shape or left.ndim!=2:
            raise ValueError('Expected same-sized monochrome stereo images')
        arrays=[np.ascontiguousarray(np.repeat(im[:,:,None],3,axis=2)) for im in (left,right)]
        tensors=[torch.from_numpy(a).permute(2,0,1)[None].to('cuda',dtype=torch.float32) for a in arrays]
        padder=self.InputPadder(tensors[0].shape,divis_by=32,force_square=False)
        a,b=padder.pad(*tensors)
        with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
            disparity=self.model.forward(a,b,iters=self.iterations,test_mode=True,optimize_build_volume='pytorch1')
        return padder.unpad(disparity.float()).reshape(left.shape).cpu().numpy()

    def warmup(self,width=384,height=256):
        rng=np.random.default_rng(481)
        left=rng.integers(0,256,(height,width),dtype=np.uint8);right=np.roll(left,-20,axis=1)
        times=[]
        for i in range(16):
            t=time.perf_counter();self.predict(left,right);elapsed=1000*(time.perf_counter()-t)
            if i>=6:times.append(elapsed)
        return dict(zip(('p50','p95','max'),map(float,np.percentile(times,[50,95,100]))))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',default=str(DEFAULT_WEIGHTS));p.add_argument('--source',default=str(DEFAULT_SOURCE))
    p.add_argument('--recording',required=True);p.add_argument('--output',required=True)
    p.add_argument('--width',type=int,default=384);p.add_argument('--height',type=int,default=256)
    p.add_argument('--iterations',type=int,default=4);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    runtime=FFSRuntime(args.weights,args.source,args.iterations)
    warmup=runtime.warmup(args.width,args.height)
    pairs=[]
    for i in (200,500,900,1300):
        with np.load(Path(args.recording)/'frames'/('ir_%04d.npz'%i)) as a:
            h,w=a['ir_left'].shape
            x=max(0,(w-args.width)//2);y=max(0,(h-args.height)//2)
            pairs.append(tuple(np.ascontiguousarray(a[k][y:y+args.height,x:x+args.width]) for k in ('ir_left','ir_right')))
    times=[]
    for i in range(24):
        left,right=pairs[i%len(pairs)]
        t=time.perf_counter();disparity=runtime.predict(left,right);times.append(1000*(time.perf_counter()-t))
    np.savez_compressed(out/'sample_disparity.npz',disparity=disparity)
    result=dict(warmup=warmup,wall_ms=dict(zip(('p50','p95','p99','max'),map(float,np.percentile(times,[50,95,99,100])))),
        shape=list(left.shape),iterations=args.iterations,checkpoint_sha256=runtime.sha256,
        torch=runtime.torch.__version__,gpu=runtime.torch.cuda.get_device_name(),
        peak_allocated_mb=runtime.torch.cuda.max_memory_allocated()/2**20,
        scope='Steady-state batch-one FP16 inference, including CPU preparation and GPU transfers. Not full pipeline latency or detection recall.')
    (out/'latency.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
