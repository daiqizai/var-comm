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
 (207,44,311.5,121.5),(178,88,203,121),(315,88,364,119),
 (445.5,47,491.5,94),(147.5,67.5,160.5,78),(360,60,370,77)]
for r in regions:p.add_redact_annot(fitz.Rect(r),fill=False)
p.apply_redactions(images=0,graphics=1,text=0)
fig=plt.figure(figsize=(W/72,H/72));ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis('off');fig.patch.set_alpha(0);ax.patch.set_alpha(0)
ax.add_patch(Rectangle((194.4,147),313.4,114.5,facecolor=bg,edgecolor='none'))
ax.add_patch(Rectangle((31.6,45.4),58.2,70,facecolor='#F7FAFC',edgecolor='none'))
for x,y,w,h,fc in [(207,44,104.5,77.5,'#FAF8FC'),(178,88,25,33,'#F7FAFC'),(315,88,49,31,'#F7FAFC'),(445.5,47,46,47,'#F7FAFC'),(147.5,67.5,13,10.5,'#F7FAFC'),(360,60,10,17,'#F7FAFC')]:
 ax.add_patch(Rectangle((x,y),w,h,facecolor=fc,edgecolor='none'))

def text(x,y,s,size=7,color=ink,ha='center',weight='normal',z=5):
 return ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',fontweight=weight,zorder=z)
def box(x,y,w,h,label=None,size=7,fc=pale,ec=line):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.8',facecolor=fc,edgecolor=ec,linewidth=.58,zorder=2))
 if label:text(x+w/2,y+h/2,label,size)
def path(points,color=blue,lw=.65,arrow=True,dash=None):
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
text(52,66,r'$f$',8.4);text(58.7,69,r'$R_s$',7.6)
box(64,67,13.7,14,r'$Q$',8.8)
path([(78,74),(85.5,74),(85.5,64.8),(88.9,64.8)],ink)
path([(79.5,72.7),(81,72.7),(81,94),(77.7,94)],ink)
box(64,89,13.7,10,r'$C_s$',8)
ax.add_patch(Circle((55.3,94),3.15,facecolor='white',edgecolor=line,linewidth=.55,zorder=3));text(55.3,94,'−',7.3)
path([(55.3,74),(55.3,90.5)],ink,.52)
path([(63.8,94),(58.8,94)],ink,.52)
path([(55.3,97.5),(55.3,104)],ink,.52)
text(57,110,r'$R_{s+1}=R_s-C_s(q_s)$',6.8)
text(85,86,r'$q_s$',6.9,blue)

# The channel is a simple horizontal route. Its frame layout is a separate
# lower inset, rather than an oversized block competing for inline width.
purple='#8B66A5';purple_line='#B6A0CA'
box(208.5,62,28,25,'LDPC\n+ QAM',6.6)
text(222.5,57,'Body',6.5,'#576A7B')
path([(201.4,74.5),(208,74.5)],ink)
path([(237,74.5),(266.8,74.5)],ink)
dot(250,74.5,purple)
ax.add_patch(Circle((271,74.5),3.3,facecolor='white',edgecolor=purple,lw=.65,zorder=4))
text(271,74.5,'+',7.3,purple)
text(271,47,'AWGN',6.5,'#576A7B')
path([(271,53),(271,70.5)],purple,.65)
text(275.5,61,r'$n$',7.2,purple)
path([(275,74.5),(281,74.5)],ink)
box(281.5,62,27.5,25,'Demap\n+ LDPC',6.6)
text(295.3,57,'Body',6.5,'#576A7B')
path([(309.5,74.5),(317.5,74.5)],ink)

# The fixed MCS controls body coding; the chosen endpoint and MCS are paid in
# the protected profile. Purple routes use separate orthogonal lanes.
box(165,102,34,18,'Protected\nprofile',6.0,fc='#F8F3FC',ec=purple_line)
path([(181,88),(181,102.3)],purple,.6,dash=(0,(3,2)))
text(191.5,96,r'$(e,a)$',7,purple)
path([(199.5,110),(201,110),(201,93),(250,93),(250,75.5)],purple,.6)
path([(190,102.5),(190,90),(222,90),(222,87.5)],purple,.6,dash=(0,(3,2)))
text(231,91,r'$a$',7,purple)

text(260,102.5,r'Frame layout ($N$ symbols)',6.0,'#576A7B')
ax.add_patch(Rectangle((212,108),7,6,facecolor='#C8B6DC',edgecolor=purple,lw=.4))
ax.add_patch(Rectangle((219,108),87,6,facecolor=rx,edgecolor=blue,lw=.4))
text(215.5,119,r'$68$',6.3,purple)
text(262.5,119,r'$N-68$',6.5,blue)

box(319,102,36,18,'Profile\ndecode',6.2,fc='#F8F3FC',ec=purple_line)
path([(278.5,74.5),(278.5,97),(315,97),(315,110),(318.5,110)],purple,.6)
text(295,94,'Noisy header',5.4,purple)
path([(333,102.5),(333,89.5),(295,89.5),(295,87.5)],purple,.6,dash=(0,(3,2)))
text(307,87.5,r'$a$',6.8,purple)
path([(349,102.5),(349,88.5)],purple,.6,dash=(0,(3,2)))
text(355.8,95,r'$e$',7,purple)
path([(355.5,110),(361.5,110),(361.5,115),(364,115)],purple,.6,dash=(0,(3,2)))
text(358,106.5,r'$e$',6.7,purple)

# Keep the overview's accumulated latent on the decoder centreline. The sum
# is centred explicitly, with all connections orthogonal.
path([(445.7,68.2),(451,68.2),(451,74.5),(458.5,74.5)],ink)
text(451.5,59.8,r'$C_s$',8.2)
ax.add_patch(Circle((463,74.5),4.0,facecolor='white',edgecolor='#7CA8C1',lw=.6,zorder=4))
text(463,74.5,r'$\sum$',6.7)
path([(467.3,74.5),(471.2,74.5)],ink)
text(469,59,r'$\hat f$',8.8,blue)
box(471.5,62,16,25,r'$D_c$',10)
path([(488,74.5),(493.5,74.5)],ink)
path([(147.7,69.5),(153,69.5),(153,74.5),(160.2,74.5)],ink)
path([(358,74.5),(364,74.5),(364,61.5),(369,61.5)],blue)

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

# Initialization has its own lane. The only iterative feedback is below the row.
path([(432,186.4),(432,204),(212,204),(212,231),(230.4,231)],blue,.6)
text(373,197.5,r'Initial state $\hat f_{m+1}$',6.8,blue,z=6)
text(260,214,'Finer-scale conditioning',6.6,blue)
text(388,214,'Residual accumulation',6.6,blue)
box(231.2,221,58,20,'Context + VAR\nargmax',6.9)
path([(290.2,231),(298.7,231)],blue,.65)
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
path([(447,231),(447,252.5),(212,252.5),(212,231),(230.4,231)],blue,.6)
text(370,260,r'$s=m+2,\ldots,10$; advance scale',6.7,blue)

overlay=io.BytesIO();fig.savefig(overlay,format='pdf',transparent=True);plt.close(fig)
ov=fitz.open(stream=overlay.getvalue(),filetype='pdf');p.show_pdf_page(p.rect,ov,0)
pdf=OUT/'NeST_Com_Fig1_thin_v5.pdf';doc.save(pdf,garbage=4,deflate=True)
d=fitz.open(pdf);pg=d[0]
for dpi,name in [(600,'NeST_Com_Fig1_thin_v5.png'),(150,'print_size.png'),(216,'review.png')]:pg.get_pixmap(dpi=dpi,alpha=False).save(OUT/name)
(OUT/'NeST_Com_Fig1_thin_v5.svg').write_text(pg.get_svg_image(text_as_path=True),encoding='utf8')
(OUT/'identity.json').write_text(json.dumps(dict(input=str(SRC),input_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),page_size_pt=[W,H],new_scientific_calls=0,vector_only=True,edit_regions=regions,output_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest()),indent=2)+'\n')
print(str(pdf))
