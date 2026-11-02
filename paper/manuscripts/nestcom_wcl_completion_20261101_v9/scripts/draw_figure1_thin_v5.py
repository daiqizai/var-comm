"""Local vector cleanup of the user's thin_v4 Fig.1; no science or image synthesis."""
from pathlib import Path
import sys,json,hashlib,io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,Circle,FancyArrowPatch
import fitz

SRC=Path(sys.argv[1]);OUT=Path(sys.argv[2]);OUT.mkdir(parents=True,exist_ok=False)
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'path','font.size':7.0,'axes.linewidth':.5})
doc=fitz.open(SRC);p=doc[0];W,H=p.rect.width,p.rect.height
ink='#283746';blue='#2477A9';orange='#AD6B18';pale='#F0F7FB';line='#AECBDD';pred='#F5D18B';rx='#80B4D0';bg='#FEFDFC'
regions=[(194.4,147,507.8,261.5),(31.6,45.4,89.8,115.4),
 (207,34,311.5,121.5),(178,88,203,121),(315,88,364,119),
 (445.5,47,491.5,94),(147.5,67.5,160.5,78),(360,60,370,77),
 (160.5,60,365,121),(145.5,65,161,80),(356.5,58.5,370.2,78)]
for r in regions:p.add_redact_annot(fitz.Rect(r),fill=False)
p.apply_redactions(images=0,graphics=1,text=0)
# Restore the complete coarse-map range label after removing only its original
# text. The isometric token grid is not redacted or covered.
p.add_redact_annot(fitz.Rect(366.5,66.7,399.5,87.3),fill=False)
p.add_redact_annot(fitz.Rect(365.5,38.5,380,51.5),fill=False)
p.add_redact_annot(fitz.Rect(378,31.5,443,42),fill=False)
p.add_redact_annot(fitz.Rect(11,199,58,245),fill=False)
p.apply_redactions(images=0,graphics=0,text=0)
fig=plt.figure(figsize=(W/72,H/72));ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis('off');fig.patch.set_alpha(0);ax.patch.set_alpha(0)
ax.add_patch(Rectangle((194.4,147),313.4,114.5,facecolor=bg,edgecolor='none'))
ax.add_patch(Rectangle((31.6,45.4),58.2,70,facecolor='#F7FAFC',edgecolor='none'))
for x,y,w,h,fc in [(207,34,104.5,87.5,'#FAF8FC'),(178,88,25,33,'#F7FAFC'),(315,88,49,31,'#F7FAFC'),(445.5,47,46,47,'#F7FAFC'),(147.5,67.5,13,10.5,'#F7FAFC'),(360,60,10,17,'#F7FAFC')]:
 ax.add_patch(Rectangle((x,y),w,h,facecolor=fc,edgecolor='none'))
# Cover the complete old outlines and detached route fragments before placing
# the replacement boxes, so their old and new edges cannot both be visible.
for x,y,w,h,fc in [(160.5,60.8,41,27.6,'#F7FAFD'),(317.5,60.8,40.5,27.7,'#F7FAFD'),(358.0,72,2.5,5,'#F7FAFD'),(358.2,43,8.3,35,'#F7FAFD')]:
 ax.add_patch(Rectangle((x,y),w,h,facecolor=fc,edgecolor='none'))
# Clear obsolete short connectors at the inter-panel gutters, then restore
# the original straight compartment borders at their exact coordinates.
for x,y,w,h in [(203.4,88,3.5,22),(312.0,87.5,3.7,23),(146.5,67,3.2,5.2)]:
 ax.add_patch(Rectangle((x,y),w,h,facecolor='white' if x>200 else '#F7FAFD',edgecolor='none'))
for x,y,w,h,fc in [(203.4,88,.488,22,'#F7FAFD'),(206.208,88,.692,22,'#FAF8FC'),(312,87.5,.405,23,'#FAF8FC'),(315.2405,87.5,.4595,23,'#F7FAFD')]:
 ax.add_patch(Rectangle((x,y),w,h,facecolor=fc,edgecolor='none'))
for x,y0,y1,c in [(203.888,88,110,'#D6E3EC'),(206.208,88,110,'#E2DCE9'),(312.405,87.5,110.5,'#E2DCE9'),(315.2405,87.5,110.5,'#D6E3EC')]:
 ax.plot([x,x],[y0,y1],color=c,lw=.5,zorder=1)

def text(x,y,s,size=7,color=ink,ha='center',weight='normal',z=5):
 return ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',fontweight=weight,zorder=z)
def box(x,y,w,h,label=None,size=7,fc=pale,ec=line):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.8',facecolor=fc,edgecolor=ec,linewidth=.58,zorder=2))
 if label:text(x+w/2,y+h/2,label,size)
routes=[]
def path(points,color=blue,lw=.65,arrow=True,dash=None):
 assert all(a[0]==b[0] or a[1]==b[1] for a,b in zip(points,points[1:])),points
 routes.append(dict(points=points,turns=len(points)-2,arrow=arrow))
 xs,ys=zip(*points);ax.plot(xs,ys,color=color,lw=lw,solid_joinstyle='round',solid_capstyle='round',linestyle=dash or '-',zorder=3)
 if arrow:
  a,b=points[-2],points[-1]
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=4.3,lw=lw,color=color,shrinkA=0,shrinkB=0,zorder=4))
def dot(x,y,color=blue):ax.add_patch(Circle((x,y),.85,facecolor=color,edgecolor='none',zorder=4))
def plus(x,y,r=3.4):
 ax.add_patch(Circle((x,y),r,facecolor='white',edgecolor=blue,linewidth=.65,zorder=4));ax.plot([x-r*.57,x+r*.57],[y,y],color=blue,lw=.6,zorder=5);ax.plot([x,x],[y-r*.57,y+r*.57],color=blue,lw=.6,zorder=5)
def grid(x,y,w,h,kind='pred',nx=7,ny=7):
 for i in range(nx):
  for j in range(ny):
   col=rx if kind=='rx' or(kind=='merge' and j*nx+i<36) else pred
   ax.add_patch(Rectangle((x+i*w/nx,y+j*h/ny),w/nx,h/ny,facecolor=col,edgecolor=blue if col==rx else orange,linewidth=.18,zorder=2))
 ax.add_patch(Rectangle((x,y),w,h,facecolor='none',edgecolor=blue if kind!='pred' else orange,linewidth=.5,zorder=3))

# Restore one residual-quantization step without a crossing feedback loop.
text(61,53,'Residual VQ',6.6)
box(37.5,66,11,16,r'$E$',9)
path([(30.5,74),(37.2,74)],ink)
path([(48.7,74),(63.8,74)],ink)
text(50.5,65.5,r'$f$',8.0);text(59,65.5,r'$R_s$',7.6)
box(64,67,13.7,14,r'$Q$',8.8)
path([(71,66.5),(71,63),(88.9,63)],ink)
path([(71,81.5),(71,88.5)],ink)
box(64,89,13.7,10,r'$C_s$',8)
ax.add_patch(Circle((55.3,94),3.15,facecolor='white',edgecolor=line,linewidth=.55,zorder=3));text(55.3,94,'−',7.3)
path([(55.3,74),(55.3,90.5)],ink,.52)
path([(63.8,94),(58.8,94)],ink,.52)
path([(55.3,97.5),(55.3,104)],ink,.52)
text(57,110,r'$R_{s+1}=R_s-C_s(q_s)$',6.8)
text(79.5,85.5,r'$q_s$',6.9,blue)

# The channel is a simple horizontal route. Its frame layout is a separate
# lower inset, rather than an oversized block competing for inline width.
purple='#8B66A5';purple_line='#B6A0CA'
box(161,62,39.5,25,'AC + fitting\nsee (b)',6.5)
box(318,62,39.5,25,'Arithmetic\ndecoder',6.5)
box(208.5,62,28,25,'LDPC\n+ QAM',6.6)
text(222.5,57,'Body',6.5,'#576A7B')
path([(201.4,74.5),(208,74.5)],ink)
path([(237,74.5),(266.8,74.5)],ink)
dot(250,74.5,purple)
ax.add_patch(Circle((271,74.5),3.3,facecolor='white',edgecolor=purple,lw=.65,zorder=4))
text(271,74.5,'+',7.3,purple)
text(271,58.5,'AWGN',6.0,'#576A7B')
path([(271,63),(271,70.5)],purple,.65)
text(276,65.5,r'$n$',7.2,purple)
path([(275,74.5),(281,74.5)],ink)
box(281.5,62,27.5,25,'Demap\n+ LDPC',6.6)
text(295.3,57,'Body',6.5,'#576A7B')
path([(309.5,74.5),(317.5,74.5)],ink)

# The fixed MCS controls body coding; the chosen endpoint and MCS are paid in
# the protected profile. Purple routes use separate orthogonal lanes.
box(207.5,102,30.5,18,'Protected\nheader',5.8,fc='#F8F3FC',ec=purple_line)
path([(181,88),(181,111),(207,111)],purple,.6,dash=(0,(3,2)))
text(194,106,r'$(e,a)$',7,purple)
path([(238.5,111),(250,111),(250,75.5)],purple,.6)
path([(222.5,101.5),(222.5,87.5)],purple,.6,dash=(0,(3,2)))
text(228.5,95,r'$a$',7,purple)

text(260,37,r'Frame layout ($N$ symbols)',6.0,'#576A7B')
ax.add_patch(Rectangle((212,42),7,5,facecolor='#C8B6DC',edgecolor=purple,lw=.4))
ax.add_patch(Rectangle((219,42),87,5,facecolor=rx,edgecolor=blue,lw=.4))
text(215.5,51,r'$68$',6.0,purple)
text(259,51,r'$N-68$',6.0,blue)

box(263,102,31,18,'Profile\ndecode',6.2,fc='#F8F3FC',ec=purple_line)
path([(278.5,74.5),(278.5,101.5)],purple,.6)
text(264,94,'Noisy\nheader',5.2,purple)
path([(288,101.5),(288,87.5)],purple,.6,dash=(0,(3,2)))
text(294.5,95,r'$a$',6.8,purple)
path([(294.5,111),(337,111),(337,88.5)],purple,.6,dash=(0,(3,2)))
text(317,106,r'$e$',7,purple)

# Keep the overview's accumulated latent on the decoder centreline. The sum
# is centred explicitly, with all connections orthogonal.
path([(445.7,74.5),(458.5,74.5)],ink)
text(451.5,59.8,r'$C_s$',8.2)
ax.add_patch(Circle((463,74.5),4.0,facecolor='white',edgecolor='#7CA8C1',lw=.6,zorder=4))
text(463,74.5,r'$\sum$',6.7)
path([(467.3,74.5),(471.2,74.5)],ink)
text(469,59,r'$\hat f$',8.8,blue)
box(471.5,62,16,25,r'$D_c$',10)
path([(488,74.5),(493.5,74.5)],ink)
path([(147.7,74.5),(160.2,74.5)],ink)
path([(358,63),(369,63)],blue)
path([(365.5,44.5),(358.2,44.5)],blue)
text(373,44.5,r'$\hat q_{<s}$',8.0,blue)
text(410.5,36.5,'Scale completion',7.1,blue)
text(376.5,75,r'$1\ldots m$',7.2,blue)
text(389.8,82,r'$m+1$',8.0,blue)

# Boundary overwrite and residual update: one horizontal reading line.
for k in [2,1,0]:
 ax.add_patch(Rectangle((199+k*1.2,174-k*1.2),10,13,facecolor='#E0EFF8',edgecolor=line,lw=.4,zorder=1))
text(204,195,r'$\hat q_{1:m}$',8.7,blue)
path([(211.7,180.7),(218.1,180.7)],ink)
box(218.5,170.2,45.5,21,'Shared VAR\nargmax',6.9)
path([(264.5,180.7),(271,180.7)],ink)
grid(272,171.8,18,18)
text(281,196,r'$q^{\mathrm{pred}}_{m+1}$',8.6,orange)
path([(291.6,180.7),(307,180.7)],ink)
grid(308.5,171.8,18,18,'merge')
grid(308.5,155.8,18,3.2,'rx',nx=9,ny=1)
text(331.8,157.5,r'$K$ received',6.2,blue,ha='left')
path([(317.5,159.8),(317.5,170.2)],blue,.6)
text(317.5,196,r'$\tilde q_{m+1}$',8.6,blue)
path([(328.2,180.7),(341.5,180.7)],ink)
box(342,172.2,31,17,r'$C_{m+1}$',9)
path([(374,180.7),(385.8,180.7)],ink)
plus(390,180.7)
text(390,158.7,r'$\hat f_m$',9.4,blue)
path([(390,164.5),(390,176.6)],blue,.6)
path([(394.2,180.7),(418,180.7)],blue)
text(432,180.2,r'$\hat f_{m+1}$',10,blue)

# Repeat the named initial state at its input port instead of routing a long
# return wire across the panel. The actual recurrence remains an explicit loop.
text(255,207,r'Init. $\hat f_{m+1}$',7.3,blue)
path([(255,213),(255,220.3)],blue,.6)
text(388,214,'Residual accumulation',6.6,blue)
box(218,221,74,20,'Context + VAR\nargmax',6.9)
path([(293,231),(298.7,231)],blue,.65)
grid(300.2,224.3,13.2,13.2)
text(306.8,245,r'$\tilde q_s$',8.3,orange)
path([(314.6,231),(324.3,231)],orange,.65)
box(325.5,221,114.5,20,r'$\hat f_s=\hat f_{s-1}+C_s(\tilde q_s)$',9.15)
path([(440.8,231),(470,231)],blue,.65)
text(455,219.2,r'$\hat f_{10}$',8.8,blue)
box(471,221,20,20,r'$D_c$',10.1)
path([(492,231),(500,231)],blue,.65)
text(503.7,231,r'$\hat x$',9.8)
dot(447,231)
path([(447,231),(447,252.5),(255,252.5),(255,242)],blue,.6)
text(255,260,'Finer-scale conditioning',6.0,blue)
text(393,260,r'$s=m+2,\ldots,10$; advance scale',6.4,blue)

# Equal baseline/size for capacity labels; the original longer Next label
# touched the capacity bar. These remain outside the bars with visible padding.
text(11.6,205.4,r'Whole $(m,0)$',6.8,ink,ha='left')
text(11.6,221.9,r'Next $(m+1,0)$',6.8,'#576A7B',ha='left')
text(11.6,238.9,r'Within $(m,K)$',6.8,blue,ha='left')

overlay=io.BytesIO();fig.savefig(overlay,format='pdf',transparent=True);plt.close(fig)
ov=fitz.open(stream=overlay.getvalue(),filetype='pdf');p.show_pdf_page(p.rect,ov,0)
pdf=OUT/'NeST_Com_Fig1_thin_v5.pdf';doc.save(pdf,garbage=4,deflate=True)
d=fitz.open(pdf);pg=d[0]
for dpi,name in [(600,'NeST_Com_Fig1_thin_v5.png'),(150,'print_size.png'),(216,'review.png')]:pg.get_pixmap(dpi=dpi,alpha=False).save(OUT/name)
(OUT/'NeST_Com_Fig1_thin_v5.svg').write_text(pg.get_svg_image(text_as_path=True),encoding='utf8')
assert all(r['turns']<=1 or r['points']==[(447,231),(447,252.5),(255,252.5),(255,242)] for r in routes)
(OUT/'identity.json').write_text(json.dumps(dict(input=str(SRC),input_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),page_size_pt=[W,H],new_scientific_calls=0,vector_only=True,edit_regions=regions,output_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),routes=routes),indent=2)+'\n')
print(str(pdf))
