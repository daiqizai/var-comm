"""Original publication with exact final-figure path and preserved failed layout."""
import pathlib
import sys
import types
E=pathlib.Path(__file__).parent.parent
sys.path.insert(0,str(E))
from a1_owner import desc
path=E/'final_publication_v1/publish.py'
assert desc(path)['sha256']=='aec83acc71aee6016078ba84a11b90b1dc2519511c03901c5b0b1ae9c5cfdae8'
source=path.read_text();assert source.count('final_figures_v1')==4
source=source.replace('final_figures_v1','final_figures_v2')
# Rendering source is in final_layout_v2; original renderer remains published too.
source=source.replace("'final_figures_v2', 'final_report_v1', 'final_publication_v1'", "'final_figures_v1', 'final_report_v1', 'final_publication_v1'",1)
module=types.ModuleType('exact_final_layout_publication');module.__file__=__file__
exec(compile(source,str(path),'exec'),module.__dict__)
module.CODE_DIRS += ['a3_metric_names_v1','final_delivery_route_v1','final_layout_v2']
module.RESULT_DIRS += ['final_figures_v1']
module.SCOPES += ['a3_metric_names_qualify_control_v1','a3_metric_names_prepare_control_v1',
                 'final_figures_initial_control_v1','final_figure_revision_initial_control_v1',
                 'final_report_initial_control_v1']
if __name__=='__main__':module.main()
