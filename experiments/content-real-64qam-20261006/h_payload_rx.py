"""Actual H receiver reconstruction and PSNR core, not a runnable GPU stage.

The caller must verify the complete CPU trace/registration/ledger closure and
admit an exclusive frozen visual runtime. This core accepts actual receiver
outcomes only; target pixels enter the separate scoring function afterwards.
No PHY decoder, asset generator, scheduler or retry is implemented here.
"""
from __future__ import annotations
from contextlib import nullcontext
import copy
import hashlib
import numpy as np
import h64_source as source_codec
import h64_phy as phy
from h64_catalog import catalogue as regenerate_catalogue, digest, token_count


class TraceIntegrityError(RuntimeError):
    """A broken evidence chain or impossible cached event is not channel loss."""


def require(value, message):
    if not value:
        raise TraceIntegrityError(message)


def receiver_view(trace):
    """Extract only the real RX event and its redundant public-profile record.

    No TX, evaluation_only, source ID, cached clean image, CPU gray flag or
    convenience token field is consulted. A missing RX field is an error.
    """
    return dict(rx=copy.deepcopy(trace['rx']),
                recorded_rx_profile=copy.deepcopy(trace['rx_profile']))


def validate_image(image):
    a = np.asarray(image)
    require(a.dtype == np.float32 and a.shape == (3, 256, 256)
            and np.isfinite(a).all() and np.all((a >= 0) & (a <= 1)),
            'Frozen renderer did not return finite float32 RGB in[0,1]')
    return a.copy()


class Receiver:
    """Deterministic receiver over an immutable public catalogue.

    provider_factory() must create a fresh RX-only prefix state. render_tokens
    receives only the decoded tokens and the *received* public m/K. Neither
    callable receives the trace, target image, source ID, TX profile or labels.
    """
    def __init__(self, catalogue, *, primitives, provider_factory, render_tokens,
                 inference_context=nullcontext):
        require(catalogue == regenerate_catalogue(catalogue['buckets']), 'Public catalogue is incomplete/modified')
        require(all(p['admission'] == 'ADMITTED' for p in catalogue['profiles']), 'Unqualified public profile')
        self.catalogue_sha256 = digest(catalogue)
        self.book = {p['profile_id']: copy.deepcopy(p) for p in catalogue['profiles']}
        require(len(self.book) == catalogue['profile_count'], 'Duplicate public profile ID')
        layouts = {b['resource_id']: b['layout'] for b in catalogue['buckets'] if b['admission'] == 'ADMITTED'}
        self.phy_keys = {pid: digest(layouts[p['resource_id']]) for pid, p in self.book.items()}
        self.primitives, self.provider_factory = primitives, provider_factory
        self.render_tokens, self.inference_context = render_tokens, inference_context

    def _wire(self, view):
        """Recheck actual decoded bits without invoking a PHY decoder."""
        require(set(view) == {'rx', 'recorded_rx_profile'}, 'Only receiver-view fields are accepted')
        rx = view['rx']; require(set(rx) == {'header', 'body', 'status'}, 'Unexpected actual RX event schema')
        h = rx['header']; require(type(h.get('header_ok')) is bool, 'Missing actual header decision')
        if not h['header_ok']:
            require(h.get('profile_id') is None and rx['body'] is None and rx['status'] == 'HEADER_REJECT'
                    and view['recorded_rx_profile'] is None, 'Rejected header trace is inconsistent')
            return dict(gray=True, reason='HEADER_REJECT', profile=None, header=h, body=None)
        require(h.get('header_crc_ok') is True and h.get('header_fields_legal') is True,
                'Accepted header lacks successful CRC/public-format decision')
        pid = h.get('profile_id'); require(type(pid) is int and pid in self.book, 'Accepted public header ID unavailable')
        p = self.book[pid]
        require(view['recorded_rx_profile'] == p, 'Cached RX profile differs from actual decoded public ID')
        b = rx['body']; require(isinstance(b, dict), 'Accepted header lacks actual body event')
        require(b.get('profile_key') == p['profile_key'] and b.get('phy_key') == self.phy_keys[pid],
                'Actual decoded body layout does not match received header')
        try:
            parsed = phy.parse_body(b['decoded_bits'], p)
        except (KeyError, ValueError, TypeError) as error:
            raise TraceIntegrityError('Actual decoded body bits are malformed') from error
        require(all(k in b and b[k] == value for k, value in parsed.items()), 'Cached body parse differs from actual decoded bits')
        require(rx['status'] == parsed['status'], 'Frame/body status differs')
        if not parsed['parser_accepted']:
            return dict(gray=True, reason=parsed.get('invalid_reason', parsed['status']), profile=p, header=h, body=parsed)
        return dict(gray=False, reason='ACTUAL_PAYLOAD_PARSED', profile=p, header=h, body=parsed)

    def reconstruct(self, view):
        """Reconstruct actual RX contents. Only protocol rejection produces gray."""
        wire = self._wire(view); p = wire['profile']; decoded = None; tokens = None
        gray = wire['gray']; reason = wire['reason']; canonical_attempted = False
        canonical = None; source_status = 'WIRE_REJECT_GRAY' if gray else None
        with self.inference_context():
            if not gray:
                if p['mode'] == 'raw':
                    tokens = phy.raw_tokens(wire['body']['payload'], p)
                    source_status = 'RAW_SOURCE_DECODED'
                else:
                    require(p['mode'] == 'arithmetic' and self.primitives is not None
                            and callable(self.provider_factory), 'Independent arithmetic receiver is unavailable')
                    canonical_attempted = True
                    try:
                        decoded = source_codec.decode_prefix(np.asarray(wire['body']['payload'], dtype=np.uint8),
                            p['m'], self.provider_factory, self.primitives)
                    except source_codec.InvalidCodeStream as error:
                        gray, canonical, reason = True, False, str(error)
                        source_status = 'ARITHMETIC_SOURCE_INVALID_GRAY'
                    # Resource/model/software errors are not caught or scored gray.
                    if not gray:
                        require(decoded.get('canonical') is True and decoded.get('status') == 'SOURCE_DECODED'
                                and decoded.get('zero_extension_reads') == 30, 'Canonical source receipt invalid')
                        tokens = np.concatenate(decoded['scales']).astype(np.int64, copy=False)
                        canonical, source_status = True, 'ARITHMETIC_SOURCE_DECODED'
                if not gray:
                    require(tokens.shape == (token_count(p['m'], p['K']),) and np.issubdtype(tokens.dtype, np.integer)
                            and np.all((tokens >= 0) & (tokens < 4096)), 'Decoded actual source token state invalid')
                    image = validate_image(self.render_tokens(tokens.copy(), p['m'], p['K']))
            if gray:
                image = np.full((3, 256, 256), .5, dtype=np.float32)
        summary = dict(status='H_ACTUAL_RX_RECONSTRUCTION_COMPLETE', source_status=source_status,
            gray=gray, reason=reason, received_profile_id=None if p is None else p['profile_id'],
            received_profile_key=None if p is None else p['profile_key'],
            received_m=None if p is None else p['m'], received_K=None if p is None else p['K'],
            received_mode=None if p is None else p['mode'],
            header_accepted=wire['header']['header_ok'],
            body_crc_accepted=None if wire['body'] is None else wire['body']['crc_accepted'],
            body_parser_accepted=None if wire['body'] is None else wire['body']['parser_accepted'],
            arithmetic_canonical_attempted=canonical_attempted, arithmetic_canonical=canonical,
            source_decode_complete=True, canonical_decode_invalid=source_status == 'ARITHMETIC_SOURCE_INVALID_GRAY',
            catalogue_sha256=self.catalogue_sha256, receiver_view_sha256=digest(view),
            image_sha256=phy.array_sha(image), actual_received_tokens_sha256=None if tokens is None else phy.array_sha(tokens),
            new_packet_decodes=0, target_image_used_for_reconstruction=False, truth_correction=False,
            cached_clean_image_used=False)
        if decoded is not None:
            summary['canonical_details'] = {k: decoded[k] for k in ('consumed_bits', 'payload_bits', 'zero_extension_reads', 'canonical', 'status')}
        return dict(image=image, received_tokens=None if tokens is None else tokens.copy(), summary=summary)


def frozen_native_receiver(native, primitives, catalogue):
    """Adapter only: caller admits/guards the exclusive bound native runtime.

    No models are constructed here and original numeric flags are not changed.
    A fresh IndependentProvider is created for every arithmetic decode call.
    """
    from h_source_driver import IndependentProvider, render_received
    return Receiver(catalogue, primitives=primitives,
        provider_factory=lambda: IndependentProvider(native, primitives),
        render_tokens=lambda tokens, m, K: render_received(native, tokens, m, K),
        inference_context=native.torch.no_grad)


def score_reconstruction(reconstruction, target_u8, *, expected_preprocessing_sha256):
    """Scoring occurs only after reconstruction; target pixels never reach RX."""
    require(reconstruction['summary'].get('status') == 'H_ACTUAL_RX_RECONSTRUCTION_COMPLETE'
            and reconstruction['summary'].get('source_decode_complete') is True, 'Pending receiver result cannot be scored')
    image = validate_image(reconstruction['image'])
    require(phy.array_sha(image) == reconstruction['summary']['image_sha256'], 'Rendered RGB changed before scoring')
    target = np.asarray(target_u8)
    require(target.dtype == np.uint8 and target.shape == (3, 256, 256), 'Original target must be registered uint8 RGB')
    require(hashlib.sha256(target.tobytes()).hexdigest() == expected_preprocessing_sha256,
            'Original target preprocessing identity changed')
    from h_source_driver import pixel_scores
    # Preserve the existing H float32 /255 target conversion and float64 MSE.
    return dict(**pixel_scores(image, target.astype(np.float32)/255.),
                target_preprocessing_sha256=expected_preprocessing_sha256,
                image_sha256=reconstruction['summary']['image_sha256'],
                metric_scope='H mean-per-image PSNR/MSE; all actual receiver failures included')


def evaluation_diagnostics(reconstruction, evaluation_only):
    """Post-reconstruction wire attribution only; cannot change the result."""
    s = reconstruction['summary']
    require(s.get('source_decode_complete') is True, 'Do not classify a pending arithmetic receiver')
    for key in ('header_correct', 'parsed_wire_matches_transmission', 'accepted_wire_mismatch'):
        require(type(evaluation_only.get(key)) is bool, 'Missing evaluation-only wire diagnostic')
    return dict(header_correct=evaluation_only['header_correct'],
        parsed_wire_matches_transmission=evaluation_only['parsed_wire_matches_transmission'],
        accepted_wire_mismatch=evaluation_only['accepted_wire_mismatch'],
        canonical_decode_invalid=s['canonical_decode_invalid'],
        reconstructed_after_accepted_wire_mismatch=bool(evaluation_only['accepted_wire_mismatch'] and not s['gray']),
        diagnostic_scope='Wire truth after RX only; not a semantic-correctness label')
