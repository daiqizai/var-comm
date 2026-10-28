# Wire contract

All frames use68 protected header symbols mapped to ±(1+j) and956 QPSK body symbols; N=1024 and energy2048. Body output n=1912 bits. Seven geometries k={478,512,544,576,608,636,637} passed actual noiseless Sionna qualification with no rate-matching repetition. Header profile catalogue has137 entries, below4096, and is public. Exact catalogue/layout identities are in the result JSONs.

Single continuous AC stream: L13+AC+padding+CRC16; source capacity k−29. S_FILL chooses the deepest legal token-wise m4..fullm6 endpoint by actual Encoder.finish length at k637. S_PROTECT retains the original single source payload and chooses the smallest qualified k from{512,544,576,608,637} that fits. No source content is gained by changing k.

SHARED_OLD concatenates the unchanged old398/119/119-bit information blocks and encodes them together with k636,n1912. Their three length/CRC fields total87 bits. No new global CRC or syndrome gate is imposed. Per-stream length, CRC and arithmetic validity decide availability.

COMPACT uses k636,n1912: three10-bit lengths plus a directory CRC16, three stream CRC16s, and common tail padding. Total paid fields94 bits; net AC capacity542. Directory and stream CRCs bind the received profile; stream CRCs additionally bind role and length. C is the complete m4 prefix; A positions[0,floor(K/2)), B positions[floor(K/2),K), K0..25. A/B probabilities depend only on actual C. K is public through the paid profile, never arbitrary uncharged lengths. Fixed-K12 and token-wise K use the same syntax.

RX checks header and fixed-position directory before using lengths. Missing/invalid C or directory gives the fixed gray image. With valid C, each enabled A/B is independently CRC/AC checked. ALL requires all enabled streams; SEQ uses consecutive enabled valid streams; SEL uses each valid stream. K1 disables A, so SEQ can use B. Tail-padding errors and whole-codeword syndrome do not add a global gate. The backend does not expose convergence status; records say unavailable rather than fabricate convergence. Accepted wrong content is audited against TX only after receiver decisions.

Independent arithmetic RX never reads TX CDFs or true source tokens. Canonical re-encoding and30-bit terminal lookahead are checked. Native cache lookup uses only recovered token content and frozen completion identity. Source identity is joined afterward for metrics. All transmitted/padding bits remain inside the fixed energy and symbol budget.
