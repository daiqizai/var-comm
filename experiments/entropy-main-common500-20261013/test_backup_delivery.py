"""Synthetic archive-admission checks; no SSH, science, real backup or deletion."""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
import backup_local as b


class BackupChecks(unittest.TestCase):
    def setUp(self):
        self.old=b.DEST
        parent=(Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/synthetic_backup_checks')
                if os.name=='posix' else Path(__file__).parent/'synthetic_backup_checks')
        parent.mkdir(parents=True,exist_ok=True)
        b.DEST=Path(tempfile.mkdtemp(prefix='case_',dir=parent))

    def tearDown(self):b.DEST=self.old

    def make(self,path='repo/results/example.csv',corrupt=False):
        original=b'full_precision_result\n0.123456789\n'
        data=original+b'changed' if corrupt else original
        manifest=dict(schema='EMAIN_E1_E2_INCREMENTAL_BACKUP_V1',scope='E1_E2',file_count=1,
                      files=[dict(path=path,bytes=len(original),sha256=hashlib.sha256(original).hexdigest())],dependencies=['prior_package'])
        metadata=json.dumps(manifest).encode();restore=b'Increment only; prior package required.\n'
        archive=b.DEST/b.ARCHIVE
        with tarfile.open(archive,'w:gz') as tar:
            for name,body in [(path,data),('backup_manifest.json',metadata),('RESTORE.md',restore)]:
                member=tarfile.TarInfo(name);member.size=len(body);tar.addfile(member,io.BytesIO(body))
        receipt=dict(status='REMOTE_E1_E2_INCREMENT_CREATED_LOCAL_VERIFICATION_PENDING',bytes=archive.stat().st_size,
                     sha256=b.sha(archive),manifest_sha256=hashlib.sha256(metadata).hexdigest(),
                     restore_sha256=hashlib.sha256(restore).hexdigest(),payload_files=1)
        (b.DEST/'backup_archive_receipt.json').write_text(json.dumps(receipt))

    def test_correct_payload_is_verified_without_duplicate_extraction(self):
        self.make();result=b.verify()
        self.assertEqual(result['payload_files_verified'],1)
        self.assertFalse((b.DEST/'repo').exists())
        self.assertEqual(result['deleted_files'],0)

    def test_corrupted_member_rejected_even_when_archive_hash_matches(self):
        self.make(corrupt=True)
        with self.assertRaisesRegex(RuntimeError,'Payload SHA failed'):b.verify()
        self.assertFalse((b.DEST/'local_verification.json').exists())

    def test_path_escape_rejected(self):
        self.make(path='../escape.csv')
        with self.assertRaisesRegex(RuntimeError,'Unsafe/duplicate'):b.verify()


if __name__=='__main__':unittest.main()
