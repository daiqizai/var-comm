# Prespecified Kodak qualitative gallery

Three four-row, five-column figures use the pre-inference protocol: source indices `[0,1,2,3]`, noise seed `2001`, N1024 and SNRs 4, 10 and 19 dB. The source population contains all 24 Kodak images; this gallery displays its fixed first four. No selection by image quality or success status is performed.

Columns: original image; SwinJSCC-80k (adapted); latent continuous JSCC; adaptive BPG + LDPC; partial digital transmission + unconditional VAR (ours). The exact internal mapping and source/archive/array hashes are retained in `display_mapping.csv`.

Inputs are the same fixed central 256×256 RGB crops, with no source resizing. The renderer reads completed float32 RGB directly, without sharpening, color adjustment or failure substitution. The images are scaled only for page layout. Swin at 19 dB is outside its training and calibration range. Matching noise labels do not claim the same received waveform.

Each figure is exported as 600 dpi PNG, PDF and SVG. Photo panels are embedded pixels in the PDF/SVG; text remains vector/editable where supported. `captions.tex` contains the three English captions. This script performs no metric scoring, channel simulation or inference.

Reproduce with the command below, changing `--out` to a new or empty directory:

```sh
/home/liulu/projects/VAR_COMM/outputs/UNIFIED-METRICS-20261002/environment/bin/python /home/liulu/projects/VAR_COMM/experiments/generalization_kodak_20261009/gallery.py --protocol results/generalization_kodak_20261009/protocol.json --source-manifest outputs/GENERALIZATION-KODAK-20261009/data/dataset_manifest.json --swin-frames outputs/GENERALIZATION-KODAK-20261009/neural_v1/SWIN80K/frames.json --p-frames outputs/GENERALIZATION-KODAK-20261009/neural_v1/P1024/frames.json --bpg-frames outputs/GENERALIZATION-KODAK-20261009/BPG_ADAPTIVE/frames.json --var-frames outputs/GENERALIZATION-KODAK-20261009/neural_v1/VAR_UNCONDITIONAL/frames.json --out paper/figures/kodak24_fixed4_N1024
```

Full frozen input identities and generated output hashes are recorded in `completion.json`. Rendering completion alone does not certify manual visual inspection; inspect the PNG files before publication.
