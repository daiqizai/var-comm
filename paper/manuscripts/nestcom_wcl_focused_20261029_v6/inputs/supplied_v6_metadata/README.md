# NeST-Com WCL V6

**NeST-Com: Fixed-Budget Next-Scale Token Communication**

V6 presents the work as a complete next-scale communication design under a hard symbol budget for each image. It identifies ARPC as progressive image compression, explicitly attributes its fixed-frame LDPC/QAM implementation to this study, and connects within-scale decodable prefixes to actual-length fitting and reconstruction.

## Deliverables

- `manuscript_doublecolumn.pdf`: five-page main letter.
- `manuscript_singlecolumn.pdf`: fifteen-page double-spaced single-column manuscript.
- `manuscript_review.docx`: synchronized editable text, native equations, and native tables.
- `supplementary.pdf`: unchanged twenty-two-page V5 supplement.
- `V6_Revision_Details_zh.md`: changes and the 4 dB figure decision.
- `Figure3_4dB_Minimal_Task_zh.md`: exact bounded task for exporting current 4 dB reconstructions.
- `V5_to_V6_changes.diff`: exact source changes for inspection.

## Figure status

**The delivered Fig. 3 remains the verified N1024/10 dB Pastries/Dog comparison.**

A 4 dB primary qualitative figure is recommended for the tighter-payload scenario, subject to inspecting its actual current-method outputs. The requested 4 dB pixels were not available in the inspected repository files. Existing timing records identify both sources' 4 dB NeST/ARPC/Latent/Swin/BPG outputs; the attached task specifies recovery and limited missing WHOLE/Ada work. No existing image has been relabelled as 4 dB.

Fig. 2 retains all 4/10/19 dB points and confidence intervals. The main text explicitly reports the existing 4 dB NeST/Ada perceptual values while retaining the 10 dB and 19 dB tradeoffs.

## Scientific identity

The parent is the delivered V5 package. All 56 pre-existing input files, all 35 figure assets, all seven table files, the complete method section, and all six display equations are unchanged. Two new audit files describe existing 4 dB record identities and asset availability. No new training, inference, channel events, metric computation, or bootstrap resampling was run.

The 88-word abstract explicitly compares the endpoint method with whole-scale transmission using shared models. The original-task VAR SOTA motivation is retained as motivation for a completion prior.

## Upload

The complete package preserves:

    paper/manuscripts/nestcom_wcl_focused_20261029_v6/

Copy that directory into the repository and upload with your usual workflow. No commit or remote update was performed.

## Rebuild

From this manuscript directory:

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    latexmk -pdf -interaction=nonstopmode -halt-on-error main_singlecolumn.tex
    latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
    python scripts/build_word.py

The main LaTeX outputs are `main.pdf` and `main_singlecolumn.pdf`; the delivered copies use the names above. Word requires Pandoc, Poppler, and python-docx.

The metadata finalizer uses the adjacent V5 directory:

    python scripts/finalize_v6_metadata.py

Review receipts are stored under `validation/`. The unchanged supplement reuses its V5 artifact identity and prior complete visual review.
