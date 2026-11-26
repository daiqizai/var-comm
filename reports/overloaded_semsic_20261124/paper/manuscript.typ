#set document(title: "Image Latents as Interference Replicas in Overloaded Multiuser Uplinks")
#set page(paper: "a4", margin: (top: 21mm, bottom: 20mm, x: 23mm),
  footer: context [#set text(size: 8.5pt, fill: rgb("#545d67"))
    Image transmission over overloaded uplinks #h(1fr) #counter(page).display()])
#set text(font: "Libertinus Serif", size: 11pt, lang: "en")
#set par(justify: true, leading: 0.56em, spacing: 0.82em)
#set heading(numbering: none)
#show heading.where(level: 1): it => block(above: 12pt, below: 5pt)[#text(size: 15pt, weight: "bold", it.body)]
#show heading.where(level: 2): it => block(above: 9pt, below: 3pt)[#text(size: 11.5pt, weight: "bold", it.body)]
#let data = json("table_cells.json")
#let rule = 0.55pt + rgb("#8a94a0")
#let cells(rows) = rows.flatten().map(v => [#v])
#text(size: 9pt, fill: rgb("#485663"))[Research draft · revised 27 November 2026]
#v(4pt)
#text(size: 25pt, weight: "bold")[Image Latents as Interference Replicas in Overloaded Multiuser Uplinks]
#v(6pt)
*Abstract.* More image transmitters than receive antennas can share the same uplink symbols. A receiver can exploit image structure when estimating the waveforms it subtracts before recovering the weakest stream. We compare linear, denoised-latent and image-domain replicas on identical simulated mixtures. At four users and two antennas, latent replicas improve weak-user PSNR over linear replicas by 2.46 and 2.06 dB at two near–far spreads. Decoding the latent to RGB and re-encoding improves PSNR further, even with the same cancellation gates. The linear replica has the lowest antenna-domain interference error. After weak-user combining, the latent replica reduces interference error relative to linear by 0.19 and 0.11 dB.

= 1. What should the receiver subtract?
In an overloaded uplink, multiple combinations of user streams can explain the same received samples. With three or four simultaneous users and two receive antennas, an image prior provides information beyond a linear spatial estimate. We study where that prior enters reception. A prior can refine a user's own reconstruction. Alternatively, it can estimate what another transmitter put on the air, allowing the receiver to subtract that contribution before reconstructing the target image.

For each resource block, the two-antenna observation has the form $bold(y) = sum_u bold(h)_u x_u + bold(n)$. Following an initial linear minimum mean-square error (LMMSE) demixing step, the receiver estimates each user's image latent. Each round forms a waveform replica $hat(x)_v$ and a blockwise reliability gate $omega_v$. The next LMMSE estimate for user $u$ uses
$ bold(y)_u = bold(y)_perp - sum_(v != u) omega_v hat(bold(h))_v hat(x)_v . $
Here $bold(y)_perp$ is the observation projected onto the data subspace. All users update in parallel. The gate also sets the residual interference variance used for the next solve. The original observation supplies the starting point in every round.

== Three-user evidence
At nominal user SNRs of 13, 8 and 4 dB, a frozen receiver with three matched LMMSE solves exposes the roles of subtraction and variance updating. Weak-user PSNR is 17.478 dB with both disabled, 20.719 with waveform subtraction alone, 16.169 with only the variance update, and 21.587 with both enabled. Updating confidence helps once the receiver has removed an interferer contribution.

Against conventional parallel interference cancellation (PIC) with damping selected on 256 validation images, the full receiver gains 3.888 dB [3.758, 4.018] on the 500-image pool. Restricting the comparison to 166 disjoint image triples gives 3.898 dB [3.658, 4.156]. Replacing the denoised replica by the linear estimate at matched gates costs 3.910 dB [3.778, 4.042]. These interventions motivate an analogous test at four users, where we also measure the error of the subtracted waveform itself.

#pagebreak()
= 2. Four users sharing two receive antennas
The four users share a frozen SwinJSCC backbone and transmit adapter. Each sends a 256 × 256 RGB image with its own Walsh signature. The four users occupy the same 12,288 complex time–frequency elements. Each of 32 blocks reserves 12 of 384 dimensions for the reference and null subspace, leaving 11,904 nonorthogonal data dimensions per user. The receiver estimates channels and activity from these in-band observations. The new experiment activates all four registered users. Each receiver performs an initial LMMSE solve plus three parallel cancellation and re-separation rounds.

We reuse the three released checkpoints and their original channel draws. The 500 images form 125 disjoint groups of four. Slot four carries the weakest user. Nominal per-user SNRs are (18, 13, 8, 4) and (20, 14, 8, 2) dB, giving 14 and 18 dB power spreads. Within each group and draw, all receivers see the exact same complex samples. We average the three paired draws within each group, then resample groups 10,000 times for pointwise 95% intervals.

== Where the replica comes from
*Linear* remodulates each user's current linear latent estimate. It inherits all gates and residual variances from the original receiver. *Denoised latent* remodulates the output of the learned image-latent denoiser. *RGB re-encoding* decodes that denoised latent to continuous RGB and applies the frozen encoder and transmit adapter to synthesize a replica. This last receiver initially uses its own reliability gates. The final image decoder and forward-link resources are shared. The RGB route adds one local decode–encode pair per user per round, or twelve per four-user reception.

#v(3pt)
#align(center)[#text(size:9.25pt)[#table(columns:(1.6fr,1fr,1fr,1fr,1fr,1fr,1fr),align:(left,center,center,center,center,center,center),
  stroke:none,inset:(x:3pt,y:3.2pt),
  table.hline(stroke:rule), [], table.cell(colspan:3)[*14 dB spread*],table.cell(colspan:3)[*18 dB spread*],
  [*Replica*],[PSNR ↑],[DISTS ↓],[NMSE ↓],[PSNR ↑],[DISTS ↓],[NMSE ↓],
  table.hline(stroke:rule), ..cells(data.absolute), table.hline(stroke:rule))]]
#text(size:9.3pt)[*Table 1.* Weak-user image quality and final interference NMSE on the 500-image pool. PSNR and NMSE use dB. NMSE compares the subtracted sum to the three true interfering data waveforms at both antennas. Only the offline scorer sees the simulation truth.]

#v(4pt)
#align(center)[#image("replica_contrasts.pdf", width: 100%)]
#text(size:9.3pt)[*Figure 1.* Direct paired differences for the denoised latent relative to two receiver alternatives. Positive values in (a) and negative values in (b) favor the latent receiver. Intervals resample the 125 image groups after pairing channel draws and weights.]

The denoised latent gains 2.464 dB [2.231, 2.700] and 2.063 dB [1.841, 2.286] over the linear replica for the weakest stream. DISTS also improves at both spreads. Yet the linear replica produces the lowest interference NMSE in each of the three rounds. The RGB route gives the highest weak-user PSNR and the largest final interference NMSE. In these receivers, the squared error of the removed waveform alone does not explain reconstructed image fidelity.

#pagebreak()
= 3. Separating the RGB representation from its gate
The RGB receiver might improve PSNR by changing how strongly it cancels each user. After observing the first three arms, we registered one follow-up on the same 750 four-user observations. We replayed all twelve original user–round gate tensors for the RGB replica, holding the removal weights and residual variances equal to those of the denoised-latent receiver. The third-round physical error of the original receiver was reproduced for every observation. The only additional image scores were for the weakest user's fixed-gate RGB reconstruction.

#v(3pt)
#align(center)[#text(size:10pt)[#table(columns:(1.18fr,1fr,1fr),align:(left,center,center),stroke:none,inset:(x:4pt,y:4pt),
  table.hline(stroke:rule),[*Fixed-gate RGB minus latent*],[*14 dB spread*],[*18 dB spread*],
  table.hline(stroke:rule), ..cells(data.fixed), table.hline(stroke:rule))]]
#text(size:9.4pt)[*Table 2.* Pairwise contrasts on the 125 four-image groups, with pointwise 95% intervals. For DISTS and interference NMSE, a negative difference favors RGB.]

The fixed-gate RGB replica retains 0.704 and 0.398 dB of weak-user PSNR improvement. Letting its gates evolve freely changes mean weak-user PSNR by approximately 0.0002 dB at each spread. At 18 dB the RGB replica increases PSNR but worsens weak-user DISTS by 0.0080 [0.0050, 0.0111] with gates fixed. Across all four users, free-gate RGB improves mean PSNR and mean DISTS over the latent route at both spreads. These averages and the weakest-user result describe different users.

On the 24 Kodak images, the original three arms use six disjoint groups. The latent replica gains 2.626 dB [1.872, 3.442] and 2.642 dB [1.802, 3.519] over linear in weak-user PSNR. Free-gate RGB gains another 0.669 dB [0.345, 1.005] and 0.362 dB [0.032, 0.714] over latent. The small number of Kodak groups limits the precision of these estimates.

== Interpretation and relation to earlier work
Learned iterative multiuser detection has precedents in DeepSIC [1]. Xie et al. decode and re-encode the strong user's image for cooperative cancellation in a two-user CSI-conditioned uplink [2, Eq. (2)]. SwinJSCC supplies the image representation used here [3]. Our local RGB receiver brings image decode–re-encode into the same four-user parallel front end. A separately trained implementation of [2] would answer a different system-level comparison.

The three-user interventions identify the value of subtracting source-informed replicas before recovering the weak image. At four users, changing the replica also changes reconstructed image quality when the gates are fixed. The best PSNR in our comparison involves repeated RGB reconstruction. Direct latent remodulation saves twelve local decode–encode pairs per reception. The receiver rankings for PSNR, perceptual distance and waveform error expose a design choice. The measurements cover registered users, simulated fading, and the specified frozen checkpoints. The #link("https://github.com/daiqizai/var-comm/tree/main/reports/replica_routing_20261127")[companion study] reports single-frame A800 receiver latency. Over-the-air operation remains open.

#v(2pt)
#text(size:9pt)[
*References* #linebreak()
[1] N. Shlezinger, R. Fu and Y. C. Eldar. DeepSIC: Deep Soft Interference Cancellation for Multiuser MIMO Detection. #link("https://arxiv.org/abs/2002.03214")[arXiv:2002.03214]. #linebreak()
[2] B. Xie et al. Multi-user Wireless Image Semantic Transmission over MIMO Multiple Access Channels. #link("https://arxiv.org/abs/2504.07969")[arXiv:2504.07969]. #linebreak()
[3] K. Yang et al. SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding. #link("https://arxiv.org/abs/2308.09361")[arXiv:2308.09361].
]
