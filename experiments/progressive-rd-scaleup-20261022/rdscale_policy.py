"""Pure selection, accounting and predeclared objective; no scientific side effects."""
import hashlib
ARMS=('old_recipe_control','multiprefix_unconditional','multiprefix_full_anchor')
SEEDS=(2026102201,2026102202)
ENDPOINTS=((7,0),(7,25),(8,0),(8,84),(8,126),(9,0),(9,128),(10,0))
def choose_validation(ids,excluded,count=200):
    assert len(ids)==len(set(ids)) and set(excluded)<=set(ids)
    candidates=[i for i,s in enumerate(ids)if s not in set(excluded)]
    candidates.sort(key=lambda i:hashlib.sha256(('progressive-rd-scaleup-validation-v1|'+ids[i]).encode()).hexdigest())
    assert len(candidates)>=count
    return candidates[:count]
def job_name(arm,seed):
    assert arm in ARMS and seed in SEEDS
    return arm+'_s'+str(seed)
def objective(prefix_mse,prefix_lpips,full_prediction=None,teacher_prediction=None):
    loss=prefix_mse+.1*prefix_lpips
    if full_prediction is None:
        assert teacher_prediction is None
        return loss,None
    assert teacher_prediction is not None
    anchor=(full_prediction-teacher_prediction.detach()).square().mean()
    return loss+anchor,anchor
