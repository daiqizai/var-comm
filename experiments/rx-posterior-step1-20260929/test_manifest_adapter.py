import unittest
from resume_n3060_manifest import validate_delta,delivery_chain,legacy
class ManifestDeltaTests(unittest.TestCase):
 def test_added_probe_only(self):
  self.assertEqual(validate_delta({'/old.py'},{'/old.py','/rx/new.py'},'/rx/'),['/rx/new.py'])
 def test_disappeared_or_new_runtime_dependency_rejected(self):
  for current in [{'/rx/new.py'},{'/old.py','/other/new.py'}]:
   with self.assertRaises(RuntimeError):validate_delta({'/old.py'},current,'/rx/')
 def test_original_stage_directory_preserved(self):
  self.assertEqual(delivery_chain.CHAIN,legacy.BASE/'delivery_chain_v1')
if __name__=='__main__':unittest.main()
