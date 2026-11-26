"""Verify the pinned source snapshot and add two opt-in receive-side observation points."""
from pathlib import Path
import argparse
import hashlib
import json

def digest(data):
    return hashlib.sha256(data).hexdigest()

def replace_one(source, before, after):
    if source.count(before) != 1:
        raise ValueError("The registered upstream source differs at " + before[:120])
    return source.replace(before, after)

def prepare(src, out, protocol):
    spec = json.loads(protocol.read_text())
    manifest = (src / "snapshot.json").read_bytes()
    assert digest(manifest) == spec["parent_snapshot_sha256"]
    entries = json.loads(manifest)["files"]
    out.mkdir(exist_ok=False)
    new = {}
    for name, info in entries.items():
        original = (src / name).read_bytes()
        assert digest(original) == info["deployed"]
        deployed = original
        if name.endswith("/overloaded_noma_4x2.py"):
            source = original.decode()
            anchor = "            outputs.append((ch_out, cr_db, alpha_um, sigma_post_rb, s_lin_r, s_lin_i))"
            source = replace_one(source, anchor,
                "            observer = getattr(self, 'combiner_observer', None)\n"
                "            if observer is not None:\n"
                "                observer(u, w_u, ch_out, cr_db)\n" + anchor)
            anchor = """            for u in range(U):
                ch_u, cr_u, alp_um, sigp_um, _, _ = stage_outs[u]
                z_clean_u = self.latent_prior(ch_u.u_restored, alp_um, sigp_um, cr_u)
                denoised_latents.append(z_clean_u)

                replica = z_clean_u"""
            source = replace_one(source, anchor, """            routing = getattr(self, 'replica_router', None)
            if routing is not None:
                clean_users = [self.latent_prior(ch.u_restored, alp, sig, cr)
                               for ch, cr, alp, sig, _, _ in stage_outs]
                replacements = routing(sic_round, stage_outs, clean_users, ghat_r, ghat_i, active_hat)
            for u in range(U):
                ch_u, cr_u, alp_um, sigp_um, _, _ = stage_outs[u]
                z_clean_u = (clean_users[u] if routing is not None else
                             self.latent_prior(ch_u.u_restored, alp_um, sigp_um, cr_u))
                denoised_latents.append(z_clean_u)

                replica = replacements[:, u] if routing is not None else z_clean_u""")
            deployed = source.encode()
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as f:
            f.write(deployed)
        new[name] = digest(deployed)
    (out / "manifest.json").write_text(json.dumps({"parent":digest(manifest),"files":new},indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("src","out","protocol"):
        p.add_argument(key,type=Path)
    a = p.parse_args()
    prepare(a.src,a.out,a.protocol)
