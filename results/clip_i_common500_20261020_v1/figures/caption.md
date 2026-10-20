CLIP-I on the original common500 at N=1024. CLIP-I denotes CLIP image–image
cosine similarity between the source and its actual received reconstruction,
using OpenAI CLIP ViT-L/14 with the frozen224-input floating-point preprocessing
and FP32 inference. Higher values indicate greater image-embedding similarity.
Points are source means after averaging each source's three original noise
realizations; error bars are pointwise95% intervals obtained from the existing
10000×500 source-bootstrap index bank. NeST-Com19dB uses the corrected T1-C m10
version; Latent JSCC and SwinJSCC use their final100k and200k checkpoints.
All original receiver outcomes retain their weight. Noise realizations differ
between historical method namespaces. This supplement does not select models
or policies and excludes development examples and N2048.
