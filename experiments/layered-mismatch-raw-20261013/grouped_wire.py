"""Paid two-group raw/raster wire, separate from the historical UEP experiment.

This module does not load a model, construct a backend, generate channel noise,
or launch an experiment. The owner must bind the imported historical Header,
CRC/mask and raw64 packet modules, qualify the actual layouts, and charge its
finite frame budget. Synthetic tests exercise only dependency-injected stubs.

The RX entry accepts received samples and public information only. Truth is
available exclusively to ``audit_after_receiver`` and never changes RX output.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Callable

import numpy as np

SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
PREFIX = tuple(sum(s * s for s in SIZES[:m]) for m in range(11))
N = 1024
HEADER_USES = 68
BODY_USES = N - HEADER_USES
Q = {"QPSK": 2, "16QAM": 4, "64QAM": 6}
HEADER_KEY = "uep-v1-profile12-crc16-tail6-conv-ratematch136-qpsk-Es2"
RECEIVER_RULE = "longest_contiguous_crc_accepted_prefix_or_gray"
SINGLE_KEEP_RULE = "KEEP_actual_hard_prefix_and_raster_partial_v1"
BODY_SESSION = "UEP_MISMATCH_RAW_V1_BODY"


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, name + " must be an integer")
    return value


def bits(value, length):
    a = np.asarray(value)
    require(a.shape == (length,) and np.isin(a, (0, 1)).all(), "Binary payload dimensions/range differ")
    return a.astype(np.uint8)


def indices_to_bits(value):
    a = np.asarray(value)
    require(a.ndim == 1 and np.issubdtype(a.dtype, np.integer)
            and ((a >= 0) & (a < 4096)).all(), "Integer 12-bit VQ indices required")
    return ((a.astype(np.int64)[:, None] >> np.arange(11, -1, -1)) & 1).astype(np.uint8).reshape(-1)


def bits_to_indices(value):
    a = np.asarray(value)
    require(a.ndim == 1 and len(a) % 12 == 0, "12-bit-aligned received payload required")
    a = bits(a, len(a))
    return (a.reshape(-1, 12).astype(np.int64) * (1 << np.arange(11, -1, -1))).sum(1)


def validate_profile(row):
    """Integer accounting only: not a replacement for actual backend admission."""
    require(integer(row["profile_id"], "profile_id") < 4096, "Paid ID exceeds 12 bits")
    require(row["N"] == N and row["G"] in (1, 2) and row["order"] == "raster", "Only N1024 raw raster")
    require(row["header_symbols"] == HEADER_USES and row["header_phy_key"] == HEADER_KEY,
            "Paid original header changed")
    grouped = row["G"] == 2
    require(row["receiver_rule"] in (RECEIVER_RULE, "DROP") if grouped else
            row["receiver_rule"] == SINGLE_KEEP_RULE, "G2 must DROP; original G1 KEEP mapping remains unchanged")
    m, kpart = (integer(row[key], key) for key in ("m", "K"))
    j = integer(row["j"], "j", 1) if grouped else None
    require(1 <= m <= len(SIZES) and (j is None or j <= m), "Invalid complete-scale boundary")
    require(grouped or row["j"] is None, "Single-group public format has no split")
    require((m == len(SIZES) and kpart == 0) or (m < len(SIZES) and kpart < SIZES[m] ** 2),
            "Partial boundary must be a strict prefix of the next scale")
    count = PREFIX[m] + kpart
    require(not grouped or count > PREFIX[j], "Enhancement group must be nonempty")
    groups = row["groups"]
    require(len(groups) == row["G"], "Paid CRC/LDPC group count differs")
    source_counts = [PREFIX[j], count - PREFIX[j]] if grouped else [count]
    for index, (g, tokens) in enumerate(zip(groups, source_counts)):
        q = Q.get(g["modulation"])
        require(q in (2, 4, 6), "Explicit QPSK/16QAM/64QAM mapping required")
        for key in ("source_bits", "information_bits", "transmitted_bits", "symbols"):
            integer(g[key], "group " + key, 1)
        require(g["source_bits"] == 12 * tokens and g["crc_bits"] == 16 and g["tail_bits"] == 0
                and g["information_bits"] == 12 * tokens + 16, "Raw/CRC group accounting changed")
        require(g["transmitted_bits"] == q * g["symbols"]
                and (not grouped or 6 * g["information_bits"] <= 5 * g["transmitted_bits"]), "Group length/rate exceeds scope")
        layout = g["layout"]
        require(layout["k"] == g["information_bits"] and layout["n"] == g["transmitted_bits"],
                "Group layout dimensions differ")
        for key in ("bg", "z", "k_ldpc", "k_filler", "n_cb", "n_cb_comp", "mother_bits",
                    "puncturing_bits", "shortening_bits", "decoder_config", "implementation",
                    "interleaver_sha256", "rate_matching" if "rate_matching" in layout else "puncturing_first_2Z"):
            require(key in layout, "Missing actual backend layout field: " + key)
        require(layout["layout_id"] == identity({k: v for k, v in layout.items() if k != "layout_id"}),
                "Unsealed backend layout")
        require(isinstance(g["phy_key"], str) and len(g["phy_key"]) == 64, "Physical group identity required")
        if "q" in g:
            require(g["q"] == q, "q/modulation contradiction")
    require(not grouped or Q[groups[0]["modulation"]] <= Q[groups[1]["modulation"]], "First-round q1 <= q2 scope")
    idle = integer(row["idle_symbols"], "idle_symbols")
    require(HEADER_USES + sum(g["symbols"] for g in groups) + idle == N, "Paid 1024-use budget differs")


class Catalogue:
    """Immutable public table; no per-frame transmitter selection is stored."""
    def __init__(self, entries):
        rows = copy.deepcopy(list(entries))
        require(0 < len(rows) <= 4096, "Freeze a nonempty <=4096 ID public subtable")
        for row in rows:
            validate_profile(row)
        ids = [row["profile_id"] for row in rows]
        require(len(set(ids)) == len(ids), "Public IDs collide")
        self.digest = identity(rows)
        self._rows = {pid: json.dumps(row, sort_keys=True, allow_nan=False) for pid, row in zip(ids, rows)}

    def row(self, profile_id):
        require(type(profile_id) is int and profile_id in self._rows, "Unknown public received ID")
        return json.loads(self._rows[profile_id])

    def header_map(self):
        return {str(pid): json.loads(row) for pid, row in self._rows.items()}


def serialize_groups(scales, profile):
    """TX only. Complete scales followed by the raster first K positions."""
    validate_profile(profile)
    m, kpart = profile["m"], profile["K"]
    required = m + int(kpart > 0)
    require(len(scales) >= required, "Insufficient actual source scales")
    checked = []
    for scale, size in zip(scales[:required], SIZES[:required]):
        a = np.asarray(scale)
        require(a.shape == (size * size,), "Flattened raster scale shape differs")
        indices_to_bits(a)  # Validate before any lossy integer conversion.
        checked.append(a.astype(np.int64))
    full = np.concatenate(checked[:m])
    if kpart:
        full = np.concatenate((full, checked[m][:kpart]))
    split = PREFIX[profile["j"]] if profile["G"] == 2 else None
    result = ([indices_to_bits(full[:split]), indices_to_bits(full[split:])]
              if split is not None else [indices_to_bits(full)])
    require([len(a) for a in result] == [g["source_bits"] for g in profile["groups"]], "Group source split differs")
    return result


def gray_state():
    return dict(kind="gray", rgb_fill=0.5, prefix=[], partial_values=[], m=0, K=0,
                order="raster", accepted_groups=0, accepted_token_count=0)


def accepted_state(profile, payloads, accepted):
    require(len(payloads) == len(accepted) == profile["G"], "Received group count differs")
    # Legacy G1 entries remain legitimate *formats* in this public ID domain.
    # This new grouped receiver always uses DROP, including a legal wrong G1
    # header. Old G1 KEEP baselines retain their separate raw433 RX unchanged.
    single = profile["G"] == 1
    if not bool(accepted[0]):
        return gray_state()
    both = profile["G"] == 2 and bool(accepted[1])
    m, kpart = (profile["m"], profile["K"]) if both or single else (profile["j"], 0)
    used = 2 if both else 1
    kept = np.concatenate(payloads[:used])
    values = bits_to_indices(kept)
    require(len(values) == PREFIX[m] + kpart, "Actual accepted public prefix length differs")
    prefix = [v.tolist() for v in np.split(values[:PREFIX[m]], np.cumsum([s * s for s in SIZES[:m]])[:-1])]
    return dict(kind="tokens", prefix=prefix, partial_values=values[PREFIX[m]:].tolist(), m=m, K=kpart,
                order="raster", accepted_groups=used, used_groups=used, accepted_token_count=len(values))


class ActualPhyBackend:
    """Forward to bound original Header and raw64 packet functions without edits.

    ``packet_ledger`` receives begin/done/failed for each actual TX/RX packet
    invocation. Its begin hook must reserve/charge before returning; raising there
    prevents the actual call. Failure never retries. The owner binds all modules
    and checks layout identity before any registered scientific frame.
    """
    def __init__(self, raw_backend, legacy_phy, raw64_packet, header, packet_ledger: Callable):
        require(callable(packet_ledger), "Actual packet ledger hook is mandatory")
        require(raw64_packet.Q == Q and raw64_packet.HEADER_KEY == HEADER_KEY, "Raw64 mapping/header identity differs")
        self.raw_backend = raw_backend
        self.legacy_phy = legacy_phy
        self.raw64_packet = raw64_packet
        self.header = header
        self.packet_ledger = packet_ledger
        self._verified = set()

    def verify_catalogue(self, catalog):
        """Actual plan metadata only; owner must call before registering frames."""
        for row in catalog.header_map().values():
            for g in row["groups"]:
                require(self.raw_backend.plan(g["information_bits"], g["transmitted_bits"], Q[g["modulation"]])
                        == g["layout"], "Current actual backend plan differs from public table")
        self._verified.add(catalog.digest)

    def require_catalogue(self, catalog):
        require(catalog.digest in self._verified, "Actual backend layouts have not been checked")

    def _call(self, direction, kind, counter, metadata, function):
        public = dict(direction=direction, packet=kind, public_counter=counter, **metadata)
        self.packet_ledger(dict(stage="begin", **public))
        try:
            value = function()
        except BaseException as error:
            self.packet_ledger(dict(stage="failed", exception_type=type(error).__name__, **public))
            raise
        self.packet_ledger(dict(stage="done", **public))
        return value

    def header_transmit(self, profile_id, counter):
        return self._call("tx", "header", counter, dict(profile_id=profile_id), lambda: self.header.transmit(profile_id))

    def header_receive(self, samples, true_snr, counter, mapping):
        return self._call("rx", "header", counter, dict(symbols=HEADER_USES), lambda: self.header.receive(samples, true_snr, mapping))

    def packet_transmit(self, payload, group, index, counter):
        q = Q[group["modulation"]]
        result = self._call("tx", "G" + str(index + 1), counter, dict(phy_key=group["phy_key"], q=q),
                            lambda: self.raw64_packet.transmit_packet(self.raw_backend, self.legacy_phy,
                                payload[None], group["transmitted_bits"], q, [counter], index, BODY_SESSION))
        wave, metadata = result
        return wave[0].detach().cpu().numpy().astype(np.float64), metadata

    def packet_receive(self, samples, group, index, true_snr, counter):
        q = Q[group["modulation"]]
        t = self.raw_backend.torch
        received = t.as_tensor(samples[None], dtype=t.float32, device=self.raw_backend.device)
        payload, accept, decoded = self._call("rx", "G" + str(index + 1), counter,
            dict(phy_key=group["phy_key"], q=q), lambda: self.raw64_packet.receive_packet(
                self.raw_backend, self.legacy_phy, received, group["information_bits"],
                group["transmitted_bits"], q, true_snr, [counter], index, BODY_SESSION))
        a = np.asarray(payload)
        c = np.asarray(decoded)
        ok = np.asarray(accept)
        require(a.shape == (1, group["source_bits"]) and c.shape == (1, group["information_bits"])
                and ok.shape == (1,) and ok.dtype == np.dtype(bool), "Actual decoder output shape/type differs")
        actual = bits(a[0], group["source_bits"])
        decoded_bits = bits(c[0], group["information_bits"])
        require(np.array_equal(actual, decoded_bits[:-16])
                and bool(self.legacy_phy.crc_accept_batch(decoded_bits)) == bool(ok[0]), "Actual hard bits/CRC contradiction")
        return actual, bool(ok[0])


def transmit(profile_id, payloads, public_counter, catalog, backend):
    """TX knows its selected public ID and actual payload, never true SNR."""
    integer(public_counter, "public_counter")
    backend.require_catalogue(catalog)
    profile = catalog.row(profile_id)
    require(len(payloads) == profile["G"], "TX group count differs")
    payloads = [bits(a, g["source_bits"]) for a, g in zip(payloads, profile["groups"])]
    header = np.asarray(backend.header_transmit(profile_id, public_counter), dtype=np.float64)
    require(header.shape == (HEADER_USES, 2) and np.isfinite(header).all(), "Actual paid header dimensions differ")
    waves, packets = [header], []
    for index, (payload, group) in enumerate(zip(payloads, profile["groups"])):
        wave, packet = backend.packet_transmit(payload, group, index, public_counter)
        wave = np.asarray(wave, dtype=np.float64)
        require(wave.shape == (group["symbols"], 2) and np.isfinite(wave).all(), "Actual body dimensions differ")
        waves.append(wave)
        packets.append(dict(index=index, q=Q[group["modulation"]], k=group["information_bits"],
                            n=group["transmitted_bits"], symbols=group["symbols"], actual_E=float(np.square(wave).sum()),
                            phy_key=group["phy_key"], actual_packet_metadata=packet))
    if profile["idle_symbols"]:
        waves.append(np.ones((profile["idle_symbols"], 2), dtype=np.float64))
    wave = np.concatenate(waves)
    require(wave.shape == (N, 2), "Complete paid waveform must have 1024 complex uses")
    return wave, dict(profile_id=profile_id, N=N, header_uses=HEADER_USES,
        body_uses=sum(g["symbols"] for g in profile["groups"]), idle_uses=profile["idle_symbols"],
        actual_E=float(np.square(wave).sum()), header_actual_E=float(np.square(header).sum()),
        idle_actual_E=2 * profile["idle_symbols"], constellation_average_Es=2,
        frame_actual_mean_Es=float(np.square(wave).sum()) / N, groups=packets,
        waveform_sha256=array_sha(wave), catalogue_sha256=catalog.digest, public_counter=public_counter)


def receive(noisy_samples, true_snr, public_counter, catalog, backend):
    """RX boundary: no clean waveform, TX ID, source pixels/tokens or error oracle."""
    integer(public_counter, "public_counter")
    require(math.isfinite(float(true_snr)), "Finite true SNR required")
    backend.require_catalogue(catalog)
    samples = np.asarray(noisy_samples, dtype=np.float64)
    require(samples.shape == (N, 2) and np.isfinite(samples).all(), "Complete finite noisy N1024 samples required")
    header = backend.header_receive(samples[:HEADER_USES], true_snr, public_counter, catalog.header_map())
    require(all(type(header[k]) is bool for k in ("header_crc_ok", "header_fields_legal", "header_ok")),
            "Actual header acceptance flags required")
    require(header["header_ok"] == (header["header_crc_ok"] and header["header_fields_legal"]), "Header acceptance contradiction")
    event = dict(header, groups=[], N=N, true_snr_db=float(true_snr), public_counter=public_counter,
                 received_sha256=array_sha(samples), catalogue_sha256=catalog.digest,
                 state=gray_state(), receive_status="HEAD_REJECT")
    if header["header_ok"]:
        # Wrong but legal received IDs are intentionally decoded under that format.
        profile = catalog.row(header["profile_id"])
        cursor, payloads, accepted = HEADER_USES, [], []
        for index, group in enumerate(profile["groups"]):
            stop = cursor + group["symbols"]
            payload, accept = backend.packet_receive(samples[cursor:stop], group, index, true_snr, public_counter)
            require(type(accept) is bool, "Actual CRC acceptance boolean required")
            payload = bits(payload, group["source_bits"])
            payloads.append(payload)
            accepted.append(accept)
            event["groups"].append(dict(index=index, crc_accept=accept, crc_reject=not accept,
                accepted_payload=payload.tolist() if accept else None, decoded_payload_sha256=array_sha(payload),
                source_bits=group["source_bits"], q=Q[group["modulation"]], symbols=group["symbols"], phy_key=group["phy_key"]))
            cursor = stop
        event["state"] = accepted_state(profile, payloads, accepted)
        if profile["G"] == 1:
            event["receive_status"] = "SINGLE_FORMAT_ACCEPT" if accepted[0] else "SINGLE_FORMAT_DROP_CRC_REJECT"
        else:
            event["receive_status"] = "G1_REJECT" if not accepted[0] else ("FULL_PREFIX" if accepted[1] else "BASE_ONLY")
        event["received_group_count"] = profile["G"]
        event["received_profile_metadata_rule"] = profile["receiver_rule"]
        event["receiver_rule"] = RECEIVER_RULE
    event["state_sha256"] = identity(event["state"])
    return event


def audit_after_receiver(event, tx_profile_id, tx_payloads):
    """Simulation-only oracle labels, separate return value; RX is not mutated.

    Body equality is interpretable only when the header is correct. Wrong legal
    IDs are counted separately and remain actual RX outcomes, never corrected.
    """
    correct_header = bool(event["header_ok"] and event["profile_id"] == tx_profile_id)
    truth = dict(header_correct=correct_header, header_false_accept=bool(event["header_ok"] and not correct_header),
                 body_comparison_available=correct_header, groups=[], undetected_body_errors=0 if correct_header else None)
    if correct_header:
        require(len(event["groups"]) == len(tx_payloads) in (1, 2), "Truth audit TX group count differs")
        for group, payload in zip(event["groups"], tx_payloads):
            actual = bits(payload, group["source_bits"])
            equal = group["decoded_payload_sha256"] == array_sha(actual)
            false_accept = bool(group["crc_accept"] and not equal)
            truth["groups"].append(dict(index=group["index"], payload_equal=equal,
                false_accept_after_receiver=false_accept, crc_reject=group["crc_reject"]))
            truth["undetected_body_errors"] += int(false_accept)
    return truth
