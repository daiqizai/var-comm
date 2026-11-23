"""Independently recompute all pointwise intervals from the event files and audit the resource ledger."""
from collections import defaultdict
import csv
import math
import os
from pathlib import Path
import stat
import numpy as np
from common import CKPT, PRIMARY, STUDY, TEMP, digest, path_check, read, record, table


def read_csv(p):
    with path_check(p).open() as handle:
        return list(csv.DictReader(handle))


def decode_contrast(track,label):
    if track in ("noma","pic"):
        if label=="interaction": return {"both":1,"cancel":-1,"covariance":-1,"neither":1}
        a,b=label.split("_minus_")
        return {a:1,b:-1}
    if track=="wave_test":
        return {
            "matched_spread_minus_no_spread": {"spread/spread":1,"no_spread/no_spread":-1},
            "spread_on_spread_weights":{"spread/spread":1,"spread/no_spread":-1},
            "spread_on_no_spread_weights":{"no_spread/spread":1,"no_spread/no_spread":-1},
            "train_spread_at_spread":{"spread/spread":1,"no_spread/spread":-1},
            "train_spread_at_no_spread":{"spread/no_spread":1,"no_spread/no_spread":-1},
            "interaction":{"spread/spread":1,"no_spread/spread":-1,"spread/no_spread":-1,"no_spread/no_spread":1},
        }[label]
    if track=="qam":
        if label.startswith("fixed_minus_native_dd_gain_over_"):
            ref=label.removeprefix("fixed_minus_native_dd_gain_over_")
            return {"dd/fixed":1,ref+"/fixed":-1,"dd/native":-1,ref+"/native":1}
        if label.startswith("dd_minus_"):
            ref,mode=label.removeprefix("dd_minus_").split("_")
            return {"dd/"+mode:1,ref+"/"+mode:-1}
        method=label.removesuffix("_fixed_minus_native")
        return {method+"/fixed":1,method+"/native":-1}
    if track=="rf":
        if label=="interaction_fixed_vs_frame_rms":
            return {"rf_bussgang/fixed":1,"blackbox_rf/fixed":-1,"rf_bussgang/frame_rms":-1,"blackbox_rf/frame_rms":1}
        if label.endswith("_rf_minus_blackbox"):
            mode=label[:-len("_rf_minus_blackbox")]
            return {"rf_bussgang/"+mode:1,"blackbox_rf/"+mode:-1}
        for method in ("rf_bussgang","blackbox_rf"):
            if label.startswith(method+"_"):
                a,b=label[len(method)+1:].split("_minus_")
                return {method+"/"+a:1,method+"/"+b:-1}
    if track=="feedback":
        if label=="joint_gain_history_minus_known":
            return {"joint/history":1,"rx/history":-1,"joint/known":-1,"rx/known":1}
        if label.startswith("joint_minus_rx_"):
            mode=label[len("joint_minus_rx_"):]
            return {"joint/"+mode:1,"rx/"+mode:-1}
        method,label=label.split("_",1)
        a,b=label.split("_minus_")
        return {method+"/"+a:1,method+"/"+b:-1}
    raise ValueError((track,label))


def verify_statistics(track):
    root=STUDY/track
    summaries=read_csv(root/"means.csv")
    contrasts=read_csv(root/"paired.csv")
    provided=read_csv(root/"source_means.csv")
    checks={}
    for split,n in (("common500",500),("kodak24",24)):
        frames=read_csv(root/(split+"_frames.csv"))
        tuple_study=track in ("noma","pic")
        ids=sorted({r["tuple_anchor" if tuple_study else "source_id"] for r in frames})
        # Manifest uses fixed-width source indices; Kodak names use natural numeric order.
        from common import PRIOR
        manifest=read(PRIOR/"evaluation/noma"/(split+"_source_manifest.json"))
        order=[r["source_id"] for r in manifest]
        assert len(ids)==n and set(order)==set(ids)
        idx={x:i for i,x in enumerate(order)}
        repetitions=sorted({int(r["seed"]) for r in frames})
        assert len(repetitions)==3
        cube={}
        for r in frames:
            if tuple_study: g,a,s=r["role"],r["arm"],r["tuple_anchor"]
            elif track=="rf":g,a,s=r["gain_db"],r["model"]+"/"+r["mode"],r["source_id"]
            elif track=="wave_test":g,a,s=r["kd"],r["trained"]+"/"+r["deploy"],r["source_id"]
            elif track=="qam":g,a,s="64QAM_10dB",r["method"]+"/"+r["mode"],r["source_id"]
            else:g,a,s=r["scenario"],r["model"]+"/"+r["mode"],r["source_id"]
            if (g,a) not in cube: cube[g,a]=np.full((3,n,2),np.nan)
            i,j=repetitions.index(int(r["seed"])),idx[s]
            assert np.isnan(cube[g,a][i,j]).all()
            cube[g,a][i,j]=[float(r[k]) for k in PRIMARY]
        assert all(np.isfinite(c).all() for c in cube.values())
        per={k:a.mean(axis=0) for k,a in cube.items()}
        if tuple_study:
            for arm in {arm for _,arm in cube}:
                per["all_users",arm]=sum(per[role,arm] for role in ("User1_13dB","User2_8dB","User3_4dB"))/3
        errors=[]
        sub=[r for r in provided if r["split"]==split]
        assert len(sub)==n*len(per)
        for r in sub:
            errors.append(float(np.abs(per[r["condition"],r["arm"]][idx[r["unit"]]]-[float(r[k]) for k in PRIMARY]).max()))
        rng=np.random.default_rng(2026112308)
        draws=rng.integers(0,n,size=(5000,n))
        starts=rng.integers(0,n,size=(5000,math.ceil(n/8)))
        circular=((starts[:,:,None]+np.arange(8))%n).reshape(5000,-1)[:,:n]
        disjoint=n//3
        disjoint_draws=np.random.default_rng(2026112308).integers(0,disjoint,size=(5000,disjoint))
        for r in [*summaries,*contrasts]:
            if r["split"]!=split: continue
            g=r["condition"]
            weights=decode_contrast(track,r["contrast"]) if "contrast" in r else {r["arm"]:1}
            val=sum(mult*per[g,name][:,PRIMARY.index(r["metric"])] for name,mult in weights.items())
            ci=np.percentile(np.mean(val[draws],axis=1),[2.5,97.5])
            errors.extend(np.abs([val.mean()-float(r["mean"]),ci[0]-float(r["ci95_lo"]),ci[1]-float(r["ci95_hi"])]))
            if "block8_ci95_lo" in r:
                ci=np.percentile(np.mean(val[circular],axis=1),[2.5,97.5])
                errors.extend(np.abs(ci-[float(r[k]) for k in ("block8_ci95_lo","block8_ci95_hi")]))
            if "disjoint_ci95_lo" in r:
                assert int(r["disjoint_n"])==disjoint
                disjoint_values=val[:disjoint]
                ci=np.percentile(disjoint_values[disjoint_draws].mean(axis=1),[2.5,97.5])
                errors.extend(np.abs([disjoint_values.mean()-float(r["disjoint_mean"]),
                                      ci[0]-float(r["disjoint_ci95_lo"]),ci[1]-float(r["disjoint_ci95_hi"])]))
        assert max(errors)<1e-10,(track,split,max(errors))
        checks[split]={"frames":len(frames),"sources":n,"source_means":len(sub),"frame_sha256":digest(root/(split+"_frames.csv")),
                       "max_abs_review_difference":float(max(errors))}
    return {"splits":checks,"mean_intervals":len(summaries),"paired_intervals":len(contrasts)}


def storage():
    result={}
    for label,root in (("study",STUDY),("checkpoints",CKPT),("run",TEMP)):
        count=size=0
        for parent,dirs,files in os.walk(root,followlinks=False):
            for name in dirs+files:
                p=Path(parent)/name
                st=p.lstat()
                assert not stat.S_ISLNK(st.st_mode),p
                if not stat.S_ISDIR(st.st_mode):
                    assert stat.S_ISREG(st.st_mode) and st.st_nlink==1,p
                    count+=1
                    size+=st.st_size
        result[label]={"path":str(root),"regular_files_at_review":count,"bytes_at_review":size}
    return result


def main():
    qa={track:verify_statistics(track) for track in ("noma","pic","rf","wave_test","feedback","qam")}
    for track in ("rf","feedback","qam"):
        rows=read(STUDY/track/"evaluation.json")["native_replay"]
        errors={m:max(abs(row["new"]-row["original"]) for row in rows if row["metric"]==m) for m in PRIMARY}
        qa[track]["historical_means_max_abs"]=errors
        qa[track]["within_replay_tolerance"]=errors["psnr_db"]<.005 and errors["dists"]<.0002
    qa["wave_test"]["training_steps"]={}
    for tag in ("spread","no_spread"):
        history=read_csv(STUDY/"training"/tag/("history_"+tag+".csv"))
        steps=[int(z["step"]) for z in history]
        assert steps==list(range(0,7001,700))
        qa["wave_test"]["training_steps"][tag]=steps[-1]
    qa["noma"]["native_first_batch_max_abs"]=max(max(z["max_diff"]) for z in read(STUDY/"noma/evaluation.json")["native_replay"])
    pic_eval=read(STUDY/"pic/evaluation.json")
    assert pic_eval["frozen_checkpoint"]==read(STUDY/"protocol.json")["references"]["frozen_4b"]["sha256"]
    frozen=read(STUDY/"pic/frozen_damping.json")
    validation=read_csv(STUDY/"pic/validation_sources.csv")
    grid=read(STUDY/"pic/protocol.json")["selection"]["coefficient_grid"]
    by_coefficient={}
    test_ids={r["unit"] for r in read_csv(STUDY/"pic/source_means.csv")}
    validation_manifest=read(STUDY/"rf/calibration_sources.json")
    calibration_ids={r["id"] for r in validation_manifest}
    from common import PRIOR
    test_pixels={r["chw_u8_sha256"] for split in ("common500","kodak24")
                 for r in read(PRIOR/"evaluation/noma"/(split+"_source_manifest.json"))}
    assert test_pixels.isdisjoint(r["sha256"] for r in validation_manifest)
    for c in grid:
        rows=[r for r in validation if float(r["damping"])==c]
        assert len(rows)==256 and {r["source_id"] for r in rows}==calibration_ids
        assert test_ids.isdisjoint(r["source_id"] for r in rows)
        by_coefficient[c]=np.mean([float(r["psnr_db"]) for r in rows])
        assert abs(by_coefficient[c]-frozen["validation_mean_psnr"][str(c)])<1e-10
    assert frozen["selected"]==max(grid,key=lambda c:(by_coefficient[c],-c))
    qa["pic"]["selected_damping"]=frozen["selected"]
    qa["pic"]["native_first_batch_max_abs"]=max(max(z["max_diff"]) for z in pic_eval["native_replay"])
    qam=read(STUDY/"qam/frozen_gains.json")
    for method,scale in qam["gains"].items():
        assert abs(scale**2*qam["mean_packet_es_before_gain"][method]-1.)<1e-10
    qa["qam"]["average_packet_energy_on_calibration_after_gain"]=1.0
    for split in ("common500","kodak24"):
        batches=sorted((STUDY/"qam/recovered_batches").glob(split+"_*.csv"))
        assert [r for p in batches for r in read_csv(p)]==read_csv(STUDY/"qam"/(split+"_frames.csv"))
    qa["qam"]["recovered_batches_match_complete_splits"]=True
    record(STUDY/"data_checks.json",qa)
    ledger=[]
    for name in ("qualify_cache","no_spread","spread","noma","wave_test","feedback","pic","qam","qam_recovery"):
        role="qam" if name=="qam_recovery" else name
        prefix="recovery_" if name=="qam_recovery" else ""
        start=read(STUDY/role/(prefix+"start.json"))
        done=read(STUDY/role/(prefix+"wait.json"))
        assert start["pid"]==done["pid"] and done["actual_wait"]==(1 if name=="qam" else 0)
        ledger.append(dict(stage=name,physical_gpu=start["physical_gpu"],pid=done["pid"],start_ticks=start["start_ticks"],
                           actual_wait=done["actual_wait"],gpu_seconds=done["device_seconds"]))
    cpu={}
    for tag in ("calibrate","calibrate_cpu","test"):
        cpu["rf_"+tag]=read(STUDY/"rf"/(tag+"_wait.json"))
    cpu["feedback_channel_calibration"]=read(STUDY/"feedback/calibrate_wait.json")
    record(STUDY/"resource_ledger.json",{"gpu_workers":ledger,"total_gpu_device_hours":sum(r["gpu_seconds"] for r in ledger)/3600,
        "cpu_workers":cpu,"cpu_worker_process_hours":sum(r["cpu_process_s"] for r in cpu.values())/3600,
        "accounting":"GPU device occupancy summed over workers, CPU process time for the separately metered CPU science workers. Git validation, file packaging, statistics, short engineering checks and idle reservations excluded.",
        "failed_attempts":[{"stage":"initial CPU 4A calibration","result":"exit 1 during attention-mask placement, before any image, channel or decode; separate CPU-port retry succeeded."},
                           {"stage":"first 5A spread training admission","result":"gate observed the preceding 4B worker's final utilization sample; no training command launched. The subsequent admitted run trained once."},
                           {"stage":"V4 CPU weight-load preflight","result":"Vendor attention mask requested CUDA in a CPU-only check. Reused the qualified CPU placement to verify both complete parameter sets before GPU inference. No images or channels in either preflight."},
                           {"stage":"V4 historical-table lookup","result":"The sparse Git checkout lacked a promised historical CSV blob. A TLS read failed after 9000 scored common500 outputs but before they were saved. Retained calibration and the failed wait. Recovered all six inputs from objects verified against the pinned tree. Rescored the 9000 lost events and added the pending 432 Kodak events with durable batches. Both attempts are charged."}]})
    record(STUDY/"storage_audit.json",storage())
    print("Reviewed all source scores and paired intervals.",flush=True)


if __name__=="__main__":
    main()
