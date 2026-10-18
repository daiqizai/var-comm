"""Reuse only under identical metric math; selection-use label is not metric math."""
import copy
def numerical_identity(metadata):
    x=copy.deepcopy(metadata)
    x['DINO_L_metadata'].pop('metric_selection_usage')
    return x
def equivalent(old,new):
    assert numerical_identity(old)==numerical_identity(new),'Metric identity changed beyond declared selection population'
