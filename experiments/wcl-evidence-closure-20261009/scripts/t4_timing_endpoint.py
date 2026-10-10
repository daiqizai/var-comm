"""Fresh online T4 entropy endpoint and disk-free measured packet callbacks.

No automatic launch, model loading, cache lookup, or policy selection. The
execution coordinator binds the qualified PHY, frozen codec/native model,
policies, source manifest and independent timing registration before use.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
import numpy as np
import t1_phy as phy
from t1_codec_runtime import InvalidSourceStream,check_native

FIXED16=(0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95)
SNRS=(4,10,19)
WARMUPS=3
REPEATS=3
NOISE_SEED=6201

class Meter:
    def __init__(self,synchronize):self.synchronize=synchronize;self.seconds={}
    def call(self,name,callback):
        self.synchronize();started=time.perf_counter()
        try:return callback()
        finally:
            self.synchronize();elapsed=time.perf_counter()-started
            phy.require(np.isfinite(elapsed) and elapsed>=0,'Invalid elapsed time')
            self.seconds[name]=self.seconds.get(name,0.)+elapsed

class TimingReservations:
    """Reserve a bounded frame before timing; persist actual callbacks afterward.

    The frame's two decoder slots are durably reserved before any measured
    work. Every actual callback runs fresh; no completed event can be reused
    as an online timing measurement. No file IO occurs inside call(). A crash
    leaves a reserved unresolved frame and blocks automatic continuation.
    The complete log is in a new independent timing directory only.
    """
    def __init__(self,path,request_sha256,packet_cap):
        self.path=Path(path);phy.require(not self.path.exists(),'Fresh timing reservation log required')
        phy.require(len(request_sha256)==64 and type(packet_cap) is int and packet_cap>0,'Bound timing cap required')
        self.f=self.path.open('x',encoding='utf-8');self.cap=packet_cap
        self.reserved=0;self.actual=0;self.active=None;self.rows=[]
        self._write(dict(kind='CONFIG',request_sha256=request_sha256,packet_cap=packet_cap,
                         old_ledger_opened=False,callback_reuse=False))
    def _write(self,value):
        self.f.write(phy.canonical(value)+'\n');self.f.flush();os.fsync(self.f.fileno())
    def begin(self,frame_id):
        phy.require(self.active is None and self.reserved+2<=self.cap,'Timing frame/cap reservation invalid')
        phy.require(all(r['frame_id']!=frame_id for r in self.rows),'No automatic timing frame repeat')
        self.reserved+=2;self.active=dict(frame_id=frame_id,events=[],status='RESERVED',reserved_slots=2)
        self._write(dict(self.active))
    def call(self,event,request,callback):
        phy.require(self.active is not None,'A durable frame reservation must precede measured RX')
        phy.require(len(self.active['events'])<2,'At most paid header and single body decode')
        phy.require(not any(r['event']['event_id']==event['event_id'] for r in self.active['events']),'Repeated timed packet event')
        row=dict(event=event,request=request,status='ENTERED');self.active['events'].append(row)
        self.actual+=1
        result=callback();row['status']='COMPLETE';row['result']=result
        return result
    def finish(self):
        phy.require(self.active is not None and all(r['status']=='COMPLETE' for r in self.active['events']),'Unresolved timing callback')
        self.active.update(status='COMPLETE',actual_packet_calls=len(self.active['events']),
                           unused_reserved_slots=2-len(self.active['events']))
        self._write(self.active);self.rows.append(self.active);self.active=None
    def snapshot(self):
        return dict(reserved_packet_slots=self.reserved,actual_packet_calls=self.actual,
                    complete_frames=len(self.rows),unresolved_frames=int(self.active is not None),cap=self.cap)
    def close(self):
        if self.active is not None:self._write(dict(self.active,status='FAILED_PRESERVED'))
        self.f.close()

class EntropyEndpoint:
    """Complete tensor->waveform and observation->single-output deployment path."""
    def __init__(self,runtime,codec,native,render_received,frozen_points,family,reservations):
        phy.require(runtime.qualified and family in phy.FAMILIES,'Qualified paid PHY and registered entropy family')
        check_native(native);phy.require(codec.native is native,'Fresh RX/TX provider must use the same frozen model identity')
        phy.require(set(map(int,frozen_points))==set(SNRS),'All three frozen SNR points required')
        for p in frozen_points.values():
            phy.require(p['family']==family and p['target_m'] in (7,8,9) and p['q'] in (2,4,6)
                        and p['nominal_rate'] in phy.RATES,'Frozen entropy selection differs')
        self.runtime,self.codec,self.native=runtime,codec,native
        self.render_received,self.points,self.family=render_received,frozen_points,family
        self.reservations=reservations;self.t=native.torch
        self.source_attempts=[];self.last_tokens=None;self.last_tx=None;self.last_rx=None
    def point(self,snr):return self.points.get(str(snr),self.points.get(int(snr)))
    def counter_for(self,snr,index):return ([1,4,7,10,13,19].index(snr)*1000+index)*3
    def noise(self,wave,source_id,snr):
        return wave+phy.standard_noise(source_id,NOISE_SEED)*10**(-float(snr)/20)
    def finish(self):pass
    def visual_encode(self,pixels):
        phy.require(pixels.dtype==np.uint8 and pixels.shape==(3,256,256),'Frozen CPU uint8 input tensor required')
        t=self.t;loaded=self.native.loaded
        x=t.as_tensor(pixels[None],dtype=t.float32,device=loaded['device'])/127.5-1
        f=loaded['vae'].quant_conv(loaded['vae'].encoder(x))
        tokens=t.cat(loaded['vae'].quantize.f_to_idxBl_or_fhat(f,to_fhat=False),1)[0].cpu().numpy()
        phy.require(tokens.shape==(680,) and np.issubdtype(tokens.dtype,np.integer) and np.all((tokens>=0)&(tokens<4096)),'Actual VQ output differs')
        return tokens.astype(np.int64,copy=False)
    def choose(self,tokens,point):
        attempts=[]
        for m in range(point['target_m'],3,-1):
            # Fresh probability state for every actual attempt. All failed
            # longer-prefix encodes and fallback re-encodes are timed.
            stream=self.codec.encode(self.family,tokens,(m,))[m]
            p=next(p for p in self.runtime.profiles.values() if p['family']==self.family
                   and p['m']==m and p['q']==point['q'] and p['nominal_rate']==point['nominal_rate'])
            fits=stream['arithmetic_bits']<=p['source_capacity_bits']
            attempts.append(dict(m=m,arithmetic_bits=stream['arithmetic_bits'],fits=fits))
            if fits:return p,stream['bits'],attempts
        raise RuntimeError('m4 actual stream exceeds proven capacity; implementation/qualification error')
    def encode(self,pixels,snr,counter,meter):
        self.counter=counter
        with self.t.no_grad():
            tokens=meter.call('TX_visual_encoding',lambda:self.visual_encode(pixels))
            p,bits,attempts=meter.call('TX_probability_entropy_fallback',lambda:self.choose(tokens,self.point(snr)))
            wave,meta=meter.call('TX_header_LDPC_modulation',lambda:self.runtime.transmit(p['profile_id'],bits,counter))
        self.last_tokens=tokens;self.source_attempts=attempts;self.last_tx=meta
        return wave
    def receive(self,observed,snr,meter,event_id):
        # Only public frame counter, observed symbols and public whole catalogue
        # enter the physical receiver. No last_tx or source tokens are read.
        with self.t.no_grad():
            rx=meter.call('RX_header_LDPC_parse',lambda:self.runtime.receive(observed,snr,self.counter,
                self.runtime.profiles,self.reservations,event_id,phase='timing'))
            self.last_rx=rx;self.gray=False
            if rx['body'] is None or not rx['body']['parser_accepted']:
                self.gray=True;return np.full((3,256,256),.5,dtype=np.float32)
            p=rx['rx_profile']
            try:
                source=meter.call('RX_probability_entropy_canonical',lambda:self.codec.decode(p['family'],rx['body']['payload'],p['m']))
            except InvalidSourceStream as exc:
                self.last_rx=dict(rx,status='SOURCE_PARSE_REJECT',source_parse_reason=str(exc))
                self.gray=True;return np.full((3,256,256),.5,dtype=np.float32)
            self.last_rx=dict(rx,status='SOURCE_DECODED',source_parse='CANONICAL')
            return meter.call('RX_VAR_and_Dc',lambda:self.render_received(self.native,source['received_tokens'],p['m'],0))

def fresh_case(endpoint,pixels,snr,source_id,counter,event_id,reservations):
    """One true outer window; IO/registration are outside TX/RX and outer time."""
    reservations.begin(event_id)
    tx=Meter(endpoint.t.cuda.synchronize);rx=Meter(endpoint.t.cuda.synchronize)
    endpoint.t.cuda.synchronize();started=time.perf_counter()
    wave=tx.call('TX_total',lambda:endpoint.encode(pixels,snr,counter,tx))
    channel_started=time.perf_counter()
    observed=endpoint.noise(wave,source_id,snr)
    channel_seconds=time.perf_counter()-channel_started
    image=rx.call('RX_total',lambda:endpoint.receive(observed,snr,rx,event_id))
    endpoint.t.cuda.synchronize();outer=time.perf_counter()-started
    reservations.finish()
    phy.require(wave.shape==(1024,2) and np.isfinite(wave).all(),'Complete transmitted frame required')
    phy.require(image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all()
                and image.min()>=0 and image.max()<=1,'Single RGB output required')
    phy.require(outer+1e-6>=channel_seconds+tx.seconds['TX_total']+rx.seconds['RX_total'],'Nested timing windows differ')
    return dict(TX_seconds=tx.seconds['TX_total'],RX_seconds=rx.seconds['RX_total'],
        software_e2e_including_channel_seconds=outer,channel_seconds=channel_seconds,
        software_e2e_excluding_channel_seconds=outer-channel_seconds,
        TX_components=tx.seconds,RX_components=rx.seconds,
        waveform_sha256=phy.array_sha(wave),observation_sha256=phy.array_sha(observed),output_sha256=phy.array_sha(image),
        actual_frame_energy=float(np.square(wave).sum()),rho=float(np.square(wave).sum())/2048,
        gray=endpoint.gray,receiver_status=endpoint.last_rx['status'],
        header_accepted=endpoint.last_rx['header']['header_ok'],
        body_crc_accepted=(None if endpoint.last_rx['body'] is None else endpoint.last_rx['body'].get('crc_accepted',endpoint.last_rx['body'].get('crc_accept'))),
        actual_rx_profile=endpoint.last_rx['rx_profile'],transmitter=endpoint.last_tx,
        source_attempts=endpoint.source_attempts,online_source_cache=False,online_output_cache=False,
        actual_packet_calls=len(reservations.rows[-1]['events']))
