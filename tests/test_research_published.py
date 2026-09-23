import importlib.util
from pathlib import Path
import pytest
def test_published_research_evidence(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    if not (root/'results/review_20260923_phase2/artifact_lineage.json').exists():
        pytest.skip('research jobs have not published complete evidence yet')
    monkeypatch.syspath_prepend(str(root/'tools'))
    spec=importlib.util.spec_from_file_location('verify_research_results',root/'tools/verify_research_results.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main()
