## common500_main_provisional

Original common500, 500 sources × 3 noises at each SNR, N=1024. Existing means and source-level 95% intervals reused unchanged; no new bootstrap. Complete m10 fits calibration sources and requires comparator-scope correction before this layout is used as a final main figure.

## source_capacity_utilization

Actual arithmetic stream bits divided by available arithmetic source capacity k−29. 500 sources per point; this is source-capacity utilization, not wireless-symbol saving. Existing frozen endpoints; m10 correction pending.

## fixed16_descriptive

All original 16 development sources, native float32 images. Bars are means; black marks are medians, not confidence intervals. Raw uses channel seed6201; continuous methods seed2001; HiFi posterior23. Only Swin and HiFi share the actual RX observation. No new bootstrap or population-level superiority claim.

## fixed16_first_two

Existing native float images, all methods uniformly clipped and rounded to RGB uint8 for display only. No sharpening, recoloring, or method-specific processing. N=1024,13dB; raw6201/continuous2001/HiFi23; only Swin and HiFi same actual observation. Fixed source order, not quality selection.

## fixed16_gallery_1

Existing native float images, all methods uniformly clipped and rounded to RGB uint8 for display only. No sharpening, recoloring, or method-specific processing. N=1024,13dB; raw6201/continuous2001/HiFi23; only Swin and HiFi same actual observation. Fixed source order, not quality selection.

## fixed16_gallery_2

Existing native float images, all methods uniformly clipped and rounded to RGB uint8 for display only. No sharpening, recoloring, or method-specific processing. N=1024,13dB; raw6201/continuous2001/HiFi23; only Swin and HiFi same actual observation. Fixed source order, not quality selection.

## fixed16_gallery_3

Existing native float images, all methods uniformly clipped and rounded to RGB uint8 for display only. No sharpening, recoloring, or method-specific processing. N=1024,13dB; raw6201/continuous2001/HiFi23; only Swin and HiFi same actual observation. Fixed source order, not quality selection.

## fixed16_gallery_4

Existing native float images, all methods uniformly clipped and rounded to RGB uint8 for display only. No sharpening, recoloring, or method-specific processing. N=1024,13dB; raw6201/continuous2001/HiFi23; only Swin and HiFi same actual observation. Fixed source order, not quality selection.

## single_codeword_system

Single-codeword implementations only. Entropy TX adds VAR probability evaluation; raw TX does not. RX consumes actual observations, paid profiles and shared models; source truth is audit-only. Raw and entropy failure rules are explicitly different. Frozen raw whole19 uses idle symbols, so 956 occupied body symbols is not asserted for every raw profile.