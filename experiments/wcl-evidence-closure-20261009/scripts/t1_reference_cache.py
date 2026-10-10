"""Read-only exact receiver-state image reuse for the first32 recovery gate."""
from pathlib import Path
import numpy as np
from t1_entropy_core import OFFSETS,read,require,sha,verify
from t1_owned_source_gate import array_sha


class RawCalibrationReference:
    """Return an exact clean prefix reference or None, with sealed provenance.

    The environment request already pins the original calibration completion.
    Lookup order is original row order, never image quality or metric value.
    Only one source's image archives are held in memory at a time.
    """
    def __init__(self,environment_request,native):
        self.environment_path=Path(environment_request)
        env=read(self.environment_path)
        self.complete_pin=env['original_visual_completion']
        verify(self.complete_pin['path'],self.complete_pin['sha256'])
        self.done=read(self.complete_pin['path'])
        require(self.done['source_count']==1000 and self.done['images_scored'],
                'Original completed calibration1000 required')
        require(self.done['frozen_visual_identity']==native.loaded['identity']==env['old_visual_identity'],
                'Reference visual identity differs')
        require(self.done['numerical_runtime']==native.flags==env['old_numerical_runtime'],
                'Reference numerical settings differ')
        self.models=native.loaded['identity']['models']
        self.base=Path(self.complete_pin['path']).parent
        self.records={r['source_index']:r for r in env['records']}
        require(all(i in self.records for i in range(32)),'Original first32 source records absent')
        self.source_index=None;self.rows=[];self.archives={};self.bindings={}

    def __call__(self,index,m,tokens):
        require(type(index) is int and 0<=index<32 and m in (7,8,9),'Only registered first32 prefixes')
        tokens=np.asarray(tokens,dtype=np.int64)
        require(tokens.shape==(OFFSETS[m],),'Complete received prefix required')
        entry=self.records[index]
        if self.source_index!=index:
            cp_path=self.base/'source_checkpoints'/f'{index:04d}.json'
            rows_path=self.base/'sources'/f'{index:04d}.json'
            verify(cp_path,self.done['outputs'][str(cp_path)])
            cp=read(cp_path)
            require(cp['source_index']==index and cp['source_id']==entry['source_id'] and cp['images_scored'],
                    'Original scored source identity differs')
            require(cp['outputs'][str(rows_path)]==self.done['outputs'][str(rows_path)],
                    'Unsealed original image rows')
            verify(rows_path,cp['outputs'][str(rows_path)])
            self.rows=read(rows_path);self.archives={};self.source_index=index
            self.archive_shas={p:h for p,h in cp['outputs'].items() if p.endswith('.npz')}
            self.bindings={str(self.environment_path):sha(self.environment_path),
                self.complete_pin['path']:self.complete_pin['sha256'],
                str(cp_path):sha(cp_path),str(rows_path):sha(rows_path)}
        for row in self.rows:
            state=row['receiver_state']
            if state['kind']!='tokens' or state['m']!=m or state['K']!=0:continue
            require(state['order']=='raster' and len(state['partial_values'])==0,
                    'Whole-prefix reference syntax differs')
            flat=np.asarray([x for scale in state['prefix'] for x in scale],dtype=np.int64)
            if not np.array_equal(flat,tokens):continue
            require(row['source_id']==entry['source_id'] and row['source_index']==index,
                    'Reference row source differs')
            archive=row['image_archive']
            if archive not in self.archives:
                verify(archive,self.archive_shas[archive])
                with np.load(archive,allow_pickle=False) as z:self.archives[archive]=z['images'].copy()
            image=self.archives[archive][row['image_slot']]
            require(image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all(),
                    'Reference image layout differs')
            require(array_sha(image)==row['image_sha256'],'Reference image content differs')
            return dict(source_id=entry['source_id'],m=m,K=0,received_tokens=flat.copy(),models=self.models,
                image=image.copy(),image_sha256=row['image_sha256'],
                bindings=dict(self.bindings,**{archive:self.archive_shas[archive]}))
        return None
