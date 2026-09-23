"""Unchanged cache mathematics with signal-aware shard-boundary pause handling."""
from latent_enhancement.runtime import ResourceBusy,write_json,digest
from short_prefix import cache
from .common import OUT,register,bindings
from .evaluation_io import SafeEvaluation
class CacheSafety:
    def __init__(self):self.safe=SafeEvaluation();self.last=None
    def check(self):
        self.safe.check();self.last=self.safe.hardware;return False

def main():
    register(OUT/'C_followups/cache_runtime.json',{'bindings':bindings([__file__,cache.__file__,__import__('token_efficiency.evaluation_io',fromlist=['x']).__file__]),'change':'signal and software-thermal pause only; same original cache keys/math; incomplete shard recomputed','synthetic':False})
    cache.Safety=CacheSafety
    try:cache.main()
    except ResourceBusy as exc:print('SAFE_CACHE_SHARD_PAUSE',str(exc),flush=True);raise SystemExit(75)
if __name__=='__main__':main()
