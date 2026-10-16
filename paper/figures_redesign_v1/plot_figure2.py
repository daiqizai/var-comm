"""Render A/B from one frozen plot table. Imports no experiment or statistics code."""
from pathlib import Path
import os, json, csv, hashlib, argparse
os.environ['CUDA_VISIBLE_DEVICES']=''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output
    out.mkdir(exist_ok=False)
    S=json.loads((HERE/'style.json').read_text());M=json.loads((HERE/'methods.json').read_text());G=json.loads((HERE/'template_geometry.json').read_text())
    with (HERE/'plot_inputs/fig2_plot_data.csv').open(encoding='utf8')as f:rows=list(csv.DictReader(f))
    exports=[];checks=[]
    rc={'font.family':S['font_family'],'font.size':8,'axes.labelsize':S['axis_label_pt'],'xtick.labelsize':S['tick_pt'],'ytick.labelsize':S['tick_pt'],'legend.fontsize':S['legend_pt'],'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':S['spine_width_pt'],'axes.edgecolor':'#303030','text.color':'#202020','axes.labelcolor':'#202020','xtick.color':'#303030','ytick.color':'#303030','legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'nestcom-fig2-redesign-v1','savefig.facecolor':'white','figure.facecolor':'white','axes.facecolor':'white','axes.unicode_minus':True,'path.simplify':False}
    def handles():
      return [Line2D([],[],color=m['color'],lw=m['linewidth'],ls=m['linestyle'],marker=m['marker'],markersize=m['markersize'],markerfacecolor='white'if m['fill']=='none'else m['color'],markeredgewidth=1.0,label=m['label'])for m in M]
    def draw(ax,spec,letter):
      for m in M:
        rr=[r for r in rows if r['method_id']==m['id']and r['metric']==spec['id']]
        rr.sort(key=lambda r:int(r['snr_db']));assert [int(r['snr_db'])for r in rr]==[4,10,19]
        x=[int(r['snr_db'])for r in rr];y=[float(r['mean'])*spec['scale']for r in rr];lo=[float(r['ci_low'])*spec['scale']for r in rr];hi=[float(r['ci_high'])*spec['scale']for r in rr]
        assert all(spec['ylim'][0]<v<spec['ylim'][1] for v in lo+hi)
        caps=3.2 if m['fill']=='none'else S['errorbar_capsize_pt']
        c=ax.errorbar(x,y,yerr=[[v-l for v,l in zip(y,lo)],[h-v for v,h in zip(y,hi)]],fmt='none',ecolor=m['color'],elinewidth=S['errorbar_width_pt'],capsize=caps,capthick=1,alpha=1,zorder=3)
        if m['fill']=='none':
          for bar in c[2]:bar.set_linestyle('--')
        ax.plot(x,y,color=m['color'],linestyle=m['linestyle'],linewidth=m['linewidth'],marker=m['marker'],markersize=m['markersize'],markerfacecolor='white'if m['fill']=='none'else m['color'],markeredgecolor=m['color'],markeredgewidth=1.0,alpha=1,zorder=4 if m['fill']=='none'else 5)
      ax.set(xlim=S['xlim'],xticks=S['xticks'],ylim=spec['ylim'],yticks=spec['yticks'],xlabel='SNR (dB)',ylabel=spec['label'])
      ax.tick_params(direction='out',length=3,width=.65,pad=3)
      ax.grid(axis='y',color='#E6E6E6',linewidth=.45,zorder=0);ax.set_axisbelow(True)
      ax.set_title('('+letter+')',loc='left',fontweight='bold',fontsize=S['panel_label_pt'],pad=5)
    def save(fig,name,style,axes,legend=None):
      fig.canvas.draw();renderer=fig.canvas.get_renderer();clipped=[]
      for t in fig.findobj(Text):
        if not t.get_visible()or not t.get_text():continue
        b=t.get_window_extent(renderer)
        if b.width and b.height and (b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5):clipped.append(t.get_text())
      assert not clipped,(name,clipped)
      if legend:
        lb=legend.get_window_extent(renderer)
        assert all(not lb.overlaps(ax.get_window_extent(renderer))for ax in axes),name
      axis_records=[dict(xlim=list(map(float,ax.get_xlim())),ylim=list(map(float,ax.get_ylim())),xticks=list(map(float,ax.get_xticks())),yticks=list(map(float,ax.get_yticks())))for ax in axes]
      for ext in ['pdf','svg','png']:
        p=out/(name+'.'+ext);fig.savefig(p,dpi=S['dpi'],bbox_inches=None,pad_inches=0,facecolor='white')
        if ext=='svg':assert '<image 'not in p.read_text() and '<text 'in p.read_text()
        exports.append(dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p),width_mm=fig.get_figwidth()*25.4,height_mm=fig.get_figheight()*25.4))
      checks.append(dict(name=name,style=style,axes=axis_records,text_clipping=[],legend_outside_axes=True if legend else None,plot_data_sha256=sha(HERE/'plot_inputs/fig2_plot_data.csv')))
      plt.close(fig)
    with plt.rc_context(rc):
      for style,L in S['layouts'].items():
        w=G['mm']['TEXTWIDTH'];h=L['height_mm'];fig=plt.figure(figsize=(w/25.4,h/25.4))
        pw=(w-L['left_mm']-L['right_mm']-L['column_gap_mm'])/2
        ph=(h-L['top_mm']-L['bottom_mm']-L['row_gap_mm'])/2
        axes=[]
        for i,spec in enumerate(S['metrics']):
          row,col=divmod(i,2);left=L['left_mm']+col*(pw+L['column_gap_mm']);bottom=L['bottom_mm']+(1-row)*(ph+L['row_gap_mm'])
          ax=fig.add_axes([left/w,bottom/h,pw/w,ph/h]);draw(ax,spec,'abcd'[i]);axes.append(ax)
        hh=handles();order=[0,3,1,4,2]
        if L['legend_position']=='top':loc='upper center';anchor=(.5,1-1.8/h)
        else:loc='lower center';anchor=(.5,1.4/h)
        legend=fig.legend([hh[i]for i in order],[M[i]['label']for i in order],loc=loc,bbox_to_anchor=anchor,ncol=3,handlelength=2.6,columnspacing=1.6,handletextpad=.6,labelspacing=.65,borderaxespad=0)
        save(fig,'fig2_'+style,style,axes,legend)
        for i,spec in enumerate(S['metrics']):
          w=G['mm']['COLUMNWIDTH'];h=S['independent_panel_height_mm'];fig=plt.figure(figsize=(w/25.4,h/25.4));ax=fig.add_axes([16/w,13/h,(w-21)/w,(h-22)/h]);draw(ax,spec,'abcd'[i]);save(fig,'fig2_'+style+'_'+spec['file'],style,[ax])
    assert checks[0]['axes']==checks[5]['axes']
    result=dict(status='RENDERED_PENDING_MANUAL_VISUAL_REVIEW',matplotlib_version=matplotlib.__version__,font_family=S['font_family'],dpi=S['dpi'],source_hashes={f:sha(HERE/f)for f in ['style.json','methods.json','plot_figure2.py','plot_inputs/fig2_plot_data.csv']},new_metric_calls=0,new_bootstrap_calls=0,exports=exports,geometry_checks=checks)
    (out/'export_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],files=len(exports),same_axes_and_data=True)))
if __name__=='__main__':main()
