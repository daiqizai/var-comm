import pytest
from tools.audit_token_efficiency_references import validate_rows,resolve_archive

def grid():
 return [{'method':'natural_reference','protocol':'paid','image_id':'s','source_pixels_sha256':'pixel','snr_db':s,'seed':n,'psnr_db':20,'lpips':.2,'dino':.8,'complex_uses':4498,'actual_total_energy':8996,'data_complex_uses':4096,'metadata_complex_uses':402} for s in [1,4,7,13,19] for n in [2001,2002,2003]]

def test_natural_resources_keep_paid_metadata_and_complete_grid():
 r=grid();assert len(validate_rows(r,{'s':'pixel'}))==1
 r[0]['metadata_complex_uses']=0
 with pytest.raises(ValueError,match='metadata'):validate_rows(r,{'s':'pixel'})

def test_missing_duplicate_and_changed_source_rejected():
 r=grid()
 with pytest.raises(ValueError,match='incomplete'):validate_rows(r[:-1],{'s':'pixel'})
 with pytest.raises(ValueError,match='duplicate'):validate_rows(r+[r[0]],{'s':'pixel'})
 with pytest.raises(ValueError,match='preprocessing'):validate_rows(r,{'s':'other'})

def test_energy_and_nonfinite_metrics_rejected():
 r=grid();r[0]['actual_total_energy']=8192
 with pytest.raises(ValueError,match='N/E'):validate_rows(r,{'s':'pixel'})
 r=grid();r[0]['dino']=float('nan')
 with pytest.raises(ValueError,match='nonfinite'):validate_rows(r,{'s':'pixel'})

def test_archive_paths_cannot_escape_existing_project_outputs():
 for p in ['/etc/passwd','/workspace/projects/OTHER/images.npz','outputs/../../../private.npz']:
  with pytest.raises(ValueError):resolve_archive(p)
