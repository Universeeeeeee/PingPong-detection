"""Render recorded decisions after timing has finished; never used by detector."""
import argparse,json
from pathlib import Path
import cv2,numpy as np


def render(recording,run):
    root=Path(recording);run=Path(run);out=run/'review';out.mkdir(exist_ok=True)
    rows=json.loads((run/'frames.json').read_text());lookup={r['frame']:r for r in rows}
    labels=json.loads((root/'analysis/rgb_2d_audit/labels.json').read_text())
    source=json.loads((root/'camera.json').read_text());start=source['action_start_sensor_timestamp_s']
    writer=cv2.VideoWriter(str(out/'realtime_results.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30.,(1280,720))
    tiles=[]
    for i,g in enumerate(labels):
        with np.load(root/'frames'/g['frame']) as f:original=f['bgr'].copy()
        im=original.copy()
        row=lookup.get(g['frame']);message='DROPPED';color=(80,80,255)
        if row:
            r=row['result_2d'];uv=r.get('uv')
            if uv is not None:cv2.drawMarker(im,tuple(np.rint(uv).astype(int)),(255,255,0),cv2.MARKER_CROSS,16,2)
            if row['measured']:
                timely=row['latency_ms']<=50;color=(0,255,0) if timely else (0,160,255)
                message='3D MEASURED' if timely else '3D LATE'
                cv2.circle(im,tuple(np.rint(uv).astype(int)),14,color,2)
            else:message='NO 3D: '+row['reason']
        cv2.rectangle(im,(0,0),(1279,82),(20,20,20),-1)
        text='t=%.2f  %s'%(g['time_s'],message)
        if row:text+='  latency=%.1f ms'%row['latency_ms']
        cv2.putText(im,text,(12,26),0,.65,color,2)
        if row and row['measured']:
            cv2.putText(im,'XYZ(m): %.3f %.3f %.3f | stereo skew %.1f ms'%(*row['xyz'],row['depth_skew_ms']),
                (12,51),0,.55,(230,230,230),1)
        cv2.putText(im,'Timed-run decisions, rendered afterwards. Cyan=RGB  green=3D within 50ms  orange=late',(12,73),0,.5,(220,220,220),1)
        writer.write(im)
        if i==190:cv2.imwrite(str(out/'preview.jpg'),im)
        if row and row['measured']:
            m=row['stereo_measurement'];attempt=next(a for a in row['attempts'] if
                a.get('best') is not None and a['best']['ir_t']==m['ir_t'])
            with np.load(root/'frames'/attempt['frame']) as f:
                pieces=[cv2.resize(cv2.getRectSubPix(original,(70,70),tuple(map(float,uv))),(140,140))]
                for key,xy in (('ir_left',m['left_uv']),('ir_right',m['right_uv'])):
                    crop=cv2.getRectSubPix(f[key],(70,70),tuple(map(float,xy)))
                    crop=cv2.cvtColor(cv2.resize(crop,(140,140)),cv2.COLOR_GRAY2BGR)
                    cv2.drawMarker(crop,(70,70),(0,255,0),cv2.MARKER_CROSS,14,1);pieces.append(crop)
            tile=cv2.copyMakeBorder(np.hstack(pieces),25,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(tile,'%s Z=%.2f ncc=%.2f'%(g['frame'],row['xyz'][2],m['ncc']),(4,17),0,.45,(255,255,255),1)
            tiles.append(tile)
    writer.release()
    for k in range(0,len(tiles),18):
        chunk=tiles[k:k+18];sheet=np.zeros((165*((len(chunk)+2)//3),1260,3),np.uint8)
        for n,tile in enumerate(chunk):sheet[n//3*165:(n//3+1)*165,n%3*420:(n%3+1)*420]=tile
        cv2.imwrite(str(out/('stereo_sheet_%02d.jpg'%(k//18))),sheet)
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('recording');p.add_argument('run');a=p.parse_args();render(a.recording,a.run)
