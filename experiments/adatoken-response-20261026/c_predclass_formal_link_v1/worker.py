"""Frozen-policy formal PHY only. Neural recovery/scoring are separate gates."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
from c_predclass_frame import runtime, choose, source_projection, NULL
from plan import build_choices, events, POPULATIONS
import argparse, traceback
import numpy as np


def main(role, path, expected):
    assert sha(path) == expected
    r = read(path)
    for d in r['code']:
        checked(d)
    assert r['scope']['sha256'] == 'b41ed4361aab6606ce6263071865cf466bc5e09cfeb7b34efabd3e56da031a68'
    assert r['revision']['sha256'] == 'fb80b863dad31fa37cac9cc2e8d8ee4a7d5248071b3a73fdcc74c704517fc254'
    checked(r['scope']); checked(r['revision'])
    population=r['population'];settings=POPULATIONS[population];count=settings['sources'];total=count*9
    assert r['caps']==dict(logical_events=total,physical_events=total,packet_decodes=2*total)
    assert r['frozen_policy']['sha256']=='68d12e9b19006e8509b3ae5a62132e73044b88ad29ca041f96b436144923ae8c'
    frozen=cr(r['frozen_policy']);assert frozen['status']=='FROZEN_FROM_FULL_CAL1000'
    fw=cr(r['freeze_actual_wait']);assert type(fw['returncode'])is int and fw['returncode']==0
    assert gone(fw['process'])and gone(fw['parent_process'])
    assert not any(fw.get(k,False)for k in('timed_out','terminated','killed'))and fw.get('metadata_error')is None
    assert type(r['max_seconds']) is int and 0 < r['max_seconds'] <= 43200
    assert len(r['source_actual_waits']) == 1
    for d in r['source_actual_waits']:
        w = cr(d)
        assert type(w['returncode']) is int and w['returncode'] == 0
        assert gone(w['process']) and gone(w['parent_process'])
        assert not any(w.get(k, False) for k in ('timed_out', 'terminated', 'killed'))
        assert w.get('metadata_error') is None
    sources = cr(r['sources'])
    assert len(sources)==count and {s['source_index'] for s in sources}==set(range(count))
    catalogue = cr(r['catalogue'])
    choices = build_choices(sources,catalogue['profiles'],population,frozen['policies'])
    jobs = events(choices,population)
    assert choices == cr(r['choices']) and jobs == cr(r['jobs']) and len(jobs) == total
    out = L / role
    out.mkdir(exist_ok=False)
    os.sched_setaffinity(0, set(range(6)))
    beg = time.monotonic(); ledger = None
    save(out/'launch.json', dict(process=ident(os.getpid()), parent_process=ident(os.getppid()),
        argv=sys.argv, registration_sha256=expected))
    try:
        phy, rt = runtime(r)
        from t2_ledger import Ledger as PacketLedger
        ledger = PacketLedger(out/'packets.sqlite', expected, r['caps']['packet_decodes'])
        (out/'frames').mkdir(); (out/'observations').mkdir()
        by_source = {s['source_index']: s for s in sources}
        by_choice = {v['key']: v for v in choices}
        streams = None; current_source = None; waves = {}; rows = []; choice_audits = {}
        for j in jobs:
            assert time.monotonic()-beg < r['max_seconds']
            c = by_choice[j['choice_key']]; s = by_source[c['source_index']]
            if current_source != c['source_index']:
                with np.load(checked(s['streams']), allow_pickle=False) as z:
                    assert set(z.files) == {'p%03d' % i for i in range(1,257)}
                    streams = {k: z[k].copy() for k in z.files}
                assert all(a.dtype == np.uint8 and a.ndim == 1 and len(a) >= 2
                    and np.isin(a, (0,1)).all() for a in streams.values())
                current_source = c['source_index']; waves = {}
            if c['waveform_key'] not in waves:
                p, bits, attempts = choose(streams, catalogue['profiles'], c['q'], c['rate'], s['predicted_class'])
                assert p['profile_id'] == c['profile_id'] and p['m'] == c['prefix']
                assert len(bits) == c['container_bits'] and len(attempts) == c['length_checks']
                assert all(streams['p%03d'%v['prefix']].size == v['bits'] for v in s['lengths'])
                wave, tx = rt.transmit(p['profile_id'], bits, c['counter'])
                waves[c['waveform_key']] = (wave, tx, asha(bits))
            wave, tx, payload_sha = waves[c['waveform_key']]
            standard = phy.standard_noise(j['source_id'], j['noise_seed'])
            observed = wave + standard * 10**(-j['snr_db']/20)
            op = out/'observations'/(j['key']+'.npz')
            with op.open('xb') as f:
                np.savez(f, observed=observed); f.flush(); os.fsync(f.fileno())
            # Only the actual observation, public SNR/counter/catalogue and call ledger
            # reach PHY RX. Desired prefix/class/stream and original source do not.
            rx = rt.receive(observed, j['snr_db'], j['counter'], rt.profiles, ledger,
                j['key'], phase='predclass_'+population)
            v = dict(event=j, choice_key=c['key'], transmission=tx, TX_payload_sha256=payload_sha,
                actual_RX=rx, source_projection=source_projection(rx), observation=pin(op),
                standard_noise_sha256=asha(standard), source_truth_supplied_to_RX=False)
            fp = out/'frames'/(j['key']+'.json'); save(fp, v); rows.append(pin(fp))
            choice_audits[c['key']] = c
            if len(rows) % 180 == 0:
                print('C_FORMAL_PHY_COMMITTED', len(rows), len(jobs), flush=True)
        assert len(rows) == total and ledger.snapshot()['unresolved'] == 0
        # Post-output diagnostics never correct an actual accepted or rejected RX.
        audit = []; accepted = 0; projections = 0
        for d in rows:
            v = cr(d); rx = v['actual_RX']; body = rx['body']
            ok = bool(rx['header']['header_ok'] and body and body['crc_accepted'] and body['parser_accepted'])
            exact = ok and rx['rx_profile']['profile_id'] == v['transmission']['profile_id'] and asha(np.array(body['payload'], np.uint8)) == v['TX_payload_sha256']
            accepted += int(ok); projections += int(v['source_projection']['status'] == 'SOURCE_INPUT')
            audit.append(dict(frame=d, accepted=ok, post_output_profile_and_payload_exact=bool(exact),
                status=rx['status'], source_projection_status=v['source_projection']['status']))
        save(out/'frames.json', rows); save(out/'post_output_audit.json', audit)
        save(out/'choice_capacity_diagnostics.json', list(choice_audits.values()))
        save(out/'completion.json', dict(status='COMPLETE_PHY_AWAITING_SOURCE_RX', namespace=settings['namespace'],population=population,
            registration_sha256=expected, source_count=count, logical_events=len(rows), physical_events=len(rows),
            packet_ledger=ledger.snapshot(), accepted_frames=accepted, projected_source_frames=projections,
            fixed_gray_frames=len(rows)-projections, frames=pin(out/'frames.json'),
            audit=pin(out/'post_output_audit.json'), capacity=pin(out/'choice_capacity_diagnostics.json'),
            CUDA_initialized=rt.torch.cuda.is_initialized(), elapsed_seconds=time.monotonic()-beg,
            neural_calls=0, metric_calls=0, no_strategy_selected=False,frozen_policy=r['frozen_policy']))
    except BaseException:
        save(out/'failure.json', dict(traceback=traceback.format_exc(),
            packets=None if ledger is None else ledger.snapshot(), automatic_retry=False))
        raise
    finally:
        if ledger is not None:
            ledger.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('role')
    p.add_argument('--registration', required=True); p.add_argument('--expected-registration', required=True)
    a = p.parse_args(); main(a.role, a.registration, a.expected_registration)
