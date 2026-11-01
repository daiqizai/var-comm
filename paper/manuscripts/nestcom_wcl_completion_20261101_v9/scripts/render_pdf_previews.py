"""Render print previews from the final vector PDFs without redrawing data."""
from pathlib import Path
import os,subprocess
P=Path(__file__).resolve().parents[1]
tool=os.environ.get('PDFTOPPM','pdftoppm')
for stem in ['supp_figure_s1_information_budget','supp_figure_s2_threshold']:
    subprocess.run([tool,'-singlefile','-png','-r','150',str(P/'figures'/(stem+'.pdf')),
                    str(P/'figures'/(stem+'_print_size'))],check=True)
