def plots(out, means, paired, candidates, chosen):
    mpl_cache=Path(os.environ['MPLCONFIGDIR'])
    require(mpl_cache.is_relative_to(L) and mpl_cache.is_dir(),'private MPLCONFIGDIR under runtime required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    matplotlib.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
        'axes.labelsize':8.5, 'xtick.labelsize':7.5, 'ytick.labelsize':7.5,
        'legend.fontsize':8, 'axes.spines.top':False, 'axes.spines.right':False,
        'axes.linewidth':.65, 'svg.fonttype':'none', 'pdf.fonttype':42, 'ps.fonttype':42,
        'savefig.facecolor':'white'})
    manifest = []
    def export(fig, name, size):
        for suffix in ('pdf', 'svg', 'png'):
            p = out/(name+'.'+suffix)
            fig.savefig(p, dpi=600, facecolor='white')  # fixed page size; no tight bounding-box resizing
            manifest.append(dict(**pin(p), size_mm=size, raster_dpi=600 if suffix=='png' else None))
        return
    def data_plot(ispaired, gray=False):
        fig, axes = plt.subplots(2,4,figsize=(180/25.4,102/25.4))
        fig.subplots_adjust(left=.065,right=.99,bottom=.12,top=.87,wspace=.46,hspace=.65)
        for i,(metric,label,factor,direction) in enumerate(METRICS):
            ax=axes.flat[i]; ax.set_title(f"({chr(97+i)}) {label}",loc='left',fontsize=9,pad=5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_xticks([4,10]);ax.set_xlim(3,11);ax.set_xlabel('SNR (dB)')
            ax.grid(axis='y',color='.9',lw=.45,zorder=0)
            if ispaired:
                rr=sorted([r for r in paired if r['metric']==metric],key=lambda r:int(r['snr_db']))
                yy=[number(r['mean'])*factor for r in rr]
                lo=[number(r['ci_low'])*factor for r in rr];hi=[number(r['ci_high'])*factor for r in rr]
                color='0.15' if gray else '#3775BA'
                ax.axhline(0,color='.45',lw=.7,ls=':',zorder=1)
                ax.vlines([4,10],lo,hi,color=color,lw=1,zorder=2)
                for x,a,b in zip((4,10),lo,hi):ax.hlines([a,b],x-.12,x+.12,color=color,lw=.9,zorder=2)
                ax.plot([4,10],yy,'o-',color=color,ms=4.5,lw=1.5,zorder=3)
                paired_label='Agree. (pp)' if metric=='convnext_top1_source_prediction' else label
                ax.set_title(f"({chr(97+i)}) Δ{paired_label}",loc='left',fontsize=9,pad=5)
            else:
                for m,method in enumerate(METHODS):
                    rr=sorted([r for r in means if r['metric']==metric and r['method']==method],key=lambda r:int(r['snr_db']))
                    yy=[number(r['mean'])*factor for r in rr];lo=[number(r['ci_low'])*factor for r in rr];hi=[number(r['ci_high'])*factor for r in rr]
                    color=['.12','.45'][m] if gray else STYLE['colors'][m]
                    ax.vlines([4,10],lo,hi,color=color,lw=1,zorder=2)
                    for x,a,b in zip((4,10),lo,hi):ax.hlines([a,b],x-.12,x+.12,color=color,lw=.9,zorder=2)
                    ax.plot([4,10],yy,color=color,marker=STYLE['markers'][m],ls=STYLE['lines'][m],
                            ms=4.5,mfc=color if m==0 else 'white',mec=color,lw=STYLE['linewidth_pt'][m],zorder=3,label=LABELS[m])
            ax.margins(y=.15)
        axes.flat[7].set_visible(False)
        if ispaired:
            fig.text(.5,.96,'Direct paired difference: NeST-Com − ARPC',ha='center',fontsize=8.5)
            fig.text(.5,.925,'Positive favors NeST for PSNR, agreement, DINO, MS-SSIM; negative for LPIPS, DreamSim, DISTS.',ha='center',fontsize=7)
        else:
            handles,labels=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.99),ncol=2,frameon=False,handlelength=2.4)
        return fig
    for paired_flag,name in ((True,'direct_paired'),):
        fig=data_plot(paired_flag);export(fig,name,[180,102]);plt.close(fig)
        fig=data_plot(paired_flag,True);fig.savefig(out/(name+'_gray.png'),dpi=300);plt.close(fig)
    selected={(r['codec'],r['snr_db']):r['policy_id'] for r in chosen}
    for gray in (False,True):
        fig,axes=plt.subplots(1,4,figsize=(180/25.4,64/25.4))
        fig.subplots_adjust(left=.07,right=.99,bottom=.22,top=.77,wspace=.52)
        for i,(codec,snr) in enumerate(((0,4),(0,10),(1,4),(1,10))):
            ax=axes[i];rr=[r for r in candidates if (r['codec'],r['snr_db'])==(codec,snr)]
            color=['.15','.45'][codec] if gray else STYLE['colors'][codec]
            names=[r['policy_id'].split('_q')[1].replace('_r',' / ').replace('-','/') for r in rr]
            for j,r in enumerate(rr):
                win=r['policy_id']==selected[codec,snr]
                ax.plot(r['calibration_mean_DINO'],j,marker='o' if win else 's',mfc=color if win else 'white',
                        mec=color,ms=5 if win else 4.5,color=color,zorder=3)
            ax.set_yticks(range(len(rr)),names);ax.set_ylim(-.6,len(rr)-.4);ax.invert_yaxis()
            ax.set_xlabel('Mean DINO');ax.set_title(f"({chr(97+i)}) {'NeST-Com' if codec==0 else 'ARPC'}\n{snr} dB",loc='left',fontsize=9)
            ax.grid(axis='x',color='.9',lw=.45);ax.margins(x=.12)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        fig.text(.5,.94,'Nine calibration candidates; filled circle = saved policy',ha='center',fontsize=8)
        fig.text(.5,.04,'Row labels: QAM bits/symbol q / nominal code rate. Each point averages 1,000 sources after three noises.',ha='center',fontsize=7)
        if gray:fig.savefig(out/'calibration_gray.png',dpi=300)
        else:export(fig,'calibration',[180,64])
        plt.close(fig)
    save(out/'figure_manifest.json',dict(status='EXPORTED_AWAITING_ACTUAL_VISUAL_REVIEW',files=manifest,
        vector_pdf=True,editable_svg_text=True,draws_or_statistics_recomputed=0,style=STYLE))
