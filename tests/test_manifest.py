"""Public verification must tolerate absent licensed data, never changed inputs."""
from __future__ import annotations
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'reproducibility/generate_manifest.py'
spec = importlib.util.spec_from_file_location('manifest_under_test', MODULE)
manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest)


class ManifestVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'raw.jsonl').write_text('{"record": 1}\n')
        (self.root / 'runner.py').write_text('print("test")\n')
        self.paths = {'raw_input': ['raw.jsonl'], 'experiment_code': ['runner.py']}
        self.root_patch = patch.object(manifest, 'ROOT', self.root)
        self.input_patch = patch.object(manifest, 'INPUTS', self.paths)
        self.root_patch.start()
        self.input_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.input_patch.stop)
        self.file = self.root / 'manifest.json'
        self.file.write_text(json.dumps(manifest.build_manifest()))

    def test_strict_verification_checks_all_inputs(self):
        self.assertEqual(manifest.verify_manifest(self.file), (2, []))

    def test_strict_mode_rejects_missing_raw(self):
        (self.root / 'raw.jsonl').unlink()
        with self.assertRaisesRegex(ValueError, 'raw.jsonl'):
            manifest.verify_manifest(self.file)

    def test_public_mode_reports_absent_raw(self):
        (self.root / 'raw.jsonl').unlink()
        self.assertEqual(manifest.verify_manifest(self.file, allow_missing_raw=True), (1, ['raw.jsonl']))

    def test_public_mode_rejects_present_changed_raw(self):
        (self.root / 'raw.jsonl').write_text('{"record": 2}\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            manifest.verify_manifest(self.file, allow_missing_raw=True)

    def test_public_mode_rejects_missing_code(self):
        (self.root / 'runner.py').unlink()
        with self.assertRaisesRegex(ValueError, 'runner.py'):
            manifest.verify_manifest(self.file, allow_missing_raw=True)

    def test_public_mode_rejects_changed_code(self):
        (self.root / 'runner.py').write_text('print("edit")\n')
        with self.assertRaisesRegex(ValueError, 'runner.py'):
            manifest.verify_manifest(self.file, allow_missing_raw=True)

    def test_manifest_cannot_omit_an_input(self):
        obj=json.loads(self.file.read_text())
        obj['files'].pop()
        self.file.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, 'inventory'):
            manifest.verify_manifest(self.file, allow_missing_raw=True)

    def test_manifest_cannot_repeat_an_input(self):
        obj=json.loads(self.file.read_text())
        obj['files'].append(obj['files'][0])
        self.file.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, 'inventory'):
            manifest.verify_manifest(self.file, allow_missing_raw=True)


class ProvenanceVerificationTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('outputs_under_test', MODULE.parent/'verify_outputs.py')
        self.outputs=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.outputs)
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/'runner.py').write_text('print(1)')
        (self.root/'raw.jsonl').write_text('original')
        self.outputs.ROOT=self.root
        self.digest=self.outputs.sha256(self.root/'raw.jsonl')
        self.ledger={'metadata':{
            'script_path':'runner.py',
            'script_sha256':self.outputs.sha256(self.root/'runner.py'),
            'protocol_sha256':{},
            'source_sha256':{'raw.jsonl':self.digest},
        }}

    def test_strict_provenance_rejects_absent_data(self):
        (self.root/'raw.jsonl').unlink()
        with self.assertRaises(FileNotFoundError):
            self.outputs.verify_provenance('test',self.ledger)

    def test_public_provenance_lists_absent_data(self):
        (self.root/'raw.jsonl').unlink()
        result=self.outputs.verify_provenance('test',self.ledger,unavailable_inputs={'raw.jsonl':self.digest})
        self.assertEqual(result['unavailable_inputs'],['raw.jsonl'])
        self.assertEqual(result['verified_digests'],1)

    def test_public_provenance_rejects_changed_present_data(self):
        (self.root/'raw.jsonl').write_text('changed')
        with self.assertRaises(AssertionError):
            self.outputs.verify_provenance('test',self.ledger,unavailable_inputs={'raw.jsonl':self.digest})

    def test_public_provenance_requires_matching_recorded_hash(self):
        (self.root/'raw.jsonl').unlink()
        with self.assertRaises(AssertionError):
            self.outputs.verify_provenance('test',self.ledger,unavailable_inputs={'raw.jsonl':'0'*64})


if __name__ == '__main__':
    unittest.main()
