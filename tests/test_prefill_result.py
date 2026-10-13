"""Public prefill receipt integrity and privacy gates, without model access."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class PublicPrefillResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('export_prefill',ROOT/'scripts/export_prefill_results.py')
        assert spec is not None and spec.loader is not None
        cls.mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)
        cls.data=json.loads((ROOT/'results/prefill-screen.json').read_text())

    def test_receipt_and_generated_report_match(self):
        self.mod.validate(self.data)
        self.assertEqual(self.mod.report(self.data),(ROOT/'results/prefill-screen.md').read_text())
        self.assertEqual(len(self.data['arms']['A']['cells']),26)
        self.assertEqual(self.data['verification']['A_B_token_matches'],26)

    def test_missing_cell_rejected(self):
        data=copy.deepcopy(self.data);data['arms']['B']['cells'].pop()
        with self.assertRaisesRegex(ValueError,'cells'):self.mod.validate(data)

    def test_cached_prefix_rejected(self):
        data=copy.deepcopy(self.data);data['arms']['B']['cells'][0]['engine_stats']['cached']=1
        with self.assertRaisesRegex(ValueError,'fresh-prefix'):self.mod.validate(data)

    def test_edited_comparison_rejected(self):
        data=copy.deepcopy(self.data)
        data['comparisons']['B'][0]['ttft_seconds']['change_percent']=0
        with self.assertRaisesRegex(ValueError,'comparison'):self.mod.validate(data)

    def test_private_raw_response_field_rejected(self):
        data=copy.deepcopy(self.data);data['notes'].append({'output_text':'private-sentinel'})
        with self.assertRaisesRegex(ValueError,'Private or raw'):self.mod.validate(data)

    def test_filesystem_and_endpoint_locations_rejected(self):
        for value in ['/private/sentinel','https://private.invalid/sentinel']:
            data=copy.deepcopy(self.data);data['notes'].append(value)
            with self.assertRaisesRegex(ValueError,'Filesystem or endpoint'):self.mod.validate(data)

    def test_changed_image_hash_rejected(self):
        data=copy.deepcopy(self.data);data['publication_artifacts']['image']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'patch/image'):self.mod.validate(data)

    def test_changed_patch_hash_rejected(self):
        data=copy.deepcopy(self.data);data['publication_artifacts']['winning_B_patch']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'patch/image'):self.mod.validate(data)


if __name__=='__main__':unittest.main()
