"""Vector diagnostic: actual clean points and a clearly labelled historical proxy."""
from shared import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rows(p):
 with p.open()as f:return list(csv.DictReader(f))

def main():
 data=O/'calibration_diagnostics';out=O/'figures_calibration_v1';out.mkdir()
 inputs=[pin(data/'endpoint_means.csv'),pin(O/'cal100_prescreen_v1/strategy_proxy.csv'),pin(O/'cal100_prescreen_v1/nomination.json')]
 means=rows(data/'endpoint_means.csv');proxy=rows(O/'cal100_prescreen_v1/strategy_proxy.csv');nom=read(O/'cal100_prescreen_v1/nomination.json')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8.5,'axes.titlesize':9,'legend.fontsize':8,'axes.linewidth':.7,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
 fig,ax=plt.subplots(1,2,figsize=(180/25.4,73/25.4));fig.subplots_adjust(left=.075,right=.985,bottom=.24,top=.84,wspace=.32)
 colors=['#3775BA','#B64342'];lines=[]
 for codec,label,marker,style in [(0,'NeST-Com (entropy-coded)','o','-'),(1,'ARPC (adapted)','s','--')]:
  rr=sorted([x for x in means if int(x['codec'])==codec],key=lambda x:int(x['endpoint']))
  line,=ax[0].plot([float(x['charged_source_bits_mean'])/1000 for x in rr],[float(x['dino_mean'])for x in rr],label=label,marker=marker,ms=3.7,lw=1.5,ls=style,color=colors[codec],mfc=colors[codec]if codec==0 else'white');lines.append(line)
 ax[0].set(xlabel='Mean terminated payload (kbit)',ylabel='Mean clean DINO similarity',title='(a) Actual cal100 source endpoints',xlim=(0,9.5),ylim=(-.05,.95));ax[0].set_xticks([0,2,4,6,8])
 chosen=nom['decisions']['0_10']['possible_best'];rr=[x for x in proxy if x['policy_id']in chosen and x['codec']=='0'and x['snr_db']=='10'];rr.sort(key=lambda x:(int(x['q']),float(x['actual_rate'])))
 labels=[]
 for j,r in enumerate(rr):
  point,low,high=map(float,(r['proxy_dino'],r['proxy_lower'],r['proxy_upper']))
  ax[1].errorbar(j,point,yerr=[[point-low],[high-point]],fmt='none',ecolor=colors[0],elinewidth=1.2,capsize=3,zorder=2)
  ax[1].plot(j,point,'o',color=colors[0],ms=4.2,zorder=3)
  labels.append(('QPSK' if r['q']=='2'else'16-QAM')+'\n'+r['nominal_rate'])
 threshold=max(float(x['proxy_lower'])for x in rr);ax[1].axhline(threshold,color='#4D4D4D',ls=':',lw=1)
 ax[1].set(xlabel='Five unresolved NeST-Com policies',ylabel='Historical-proxy DINO',title='(b) 10 dB proxy bounds',ylim=(.59,.85),xlim=(-.45,4.45));ax[1].set_xticks(range(len(rr)),labels)
 for a in ax:a.grid(axis='y',lw=.4,color='#DDDDDD');a.set_axisbelow(True);a.tick_params(length=3,labelsize=7.5)
 fig.legend(lines,[l.get_label()for l in lines],loc='upper center',bbox_to_anchor=(.28,1),frameon=False,ncol=2,columnspacing=1.2,handlelength=2.5)
 fig.text(.746,.965,'Bounds are a screening proxy, not new link results',ha='center',va='top',fontsize=7.5)
 for suffix in ('pdf','svg','png'):fig.savefig(out/('cal100_diagnostics.'+suffix),dpi=600)
 for line in lines:line.set_color('#222222');line.set_markeredgecolor('#222222');line.set_markerfacecolor('#222222'if line.get_marker()=='o'else'white')
 for line in ax[1].lines:line.set_color('#222222');line.set_markeredgecolor('#222222')
 for coll in ax[1].collections:coll.set_color('#222222')
 fig.savefig(out/'cal100_diagnostics_grayscale.png',dpi=300);plt.close(fig)
 caption='Cal100 diagnostics, not final wireless-test performance. (a) Each point is the mean of 100 sources at an existing complete terminated source endpoint; NeST-Com has 25 endpoints and ARPC has seven. Payload includes source termination and byte rounding, with ARPC caption/mode fields; outer-frame overhead is listed separately in layouts.csv. Lines connect discrete measured means without interpolation or fitted observations. (b) Five NeST-Com MCS policies remain plausible at 10 dB under the frozen coarse proxy. Bars propagate historical single-packet joint-acceptance Bonferroni bounds after squaring under a conditional-independence approximation; they are not paired-bootstrap quality confidence intervals or new dual-fragment reliability measurements. The dotted line marks the largest proxy lower bound. Each policy uses its actual per-source fallback quality. No new bootstrap or model inference was used for this figure.\n'
 (out/'caption_en.txt').open('x').write(caption)
 put(out/'manifest.json',dict(status='RENDERED_VISUAL_REVIEW_REQUIRED',source=pin(__file__),inputs=inputs,files=[pin(p)for p in sorted(out.iterdir())if p.is_file()],size_mm=[180,73],font_sizes_pt=[7.5,8,8.5,9],vector_pdf_svg=True,new_science=0))
 print(json.dumps(dict(status='RENDERED_REQUIRES_ACTUAL_OPEN',output=str(out))))

if __name__=='__main__':main()
