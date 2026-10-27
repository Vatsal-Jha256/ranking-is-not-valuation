#!/usr/bin/env python3
"""Check released experiment results and source provenance without refitting models.

This checks the code and result files in a public checkout. Absent licensed data
are reported explicitly; the check does not claim to validate unavailable bytes.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from generate_manifest import verify_manifest
from verify_outputs import verify_provenance

ROOT=Path(__file__).resolve().parents[1]
LEDGERS={
    'selection_gap': ('exp_submission_harmonized_v4',24),
    'model_robustness': ('exp_model_robustness_v3',96),
    'calibration_structure': ('exp_calibration_structure_v1',24),
    'uncertainty_policies': ('exp_uncertainty_policies_v1',24),
    'candidate_sensitivity': ('exp_candidate_sensitivity_v1',360),
    'temporal_holdout': ('exp_temporal_holdout_v1',6),
    'beta_shrinkage': ('exp_beta_shrinkage_v2',30),
    'budget_allocation': ('exp_knapsack_v2',24),
    'stratified_quantile': ('exp_stratified_quantile_v1',24),
}


def main():
    checked,missing=verify_manifest(allow_missing_raw=True)
    manifest=json.loads((ROOT/'reproducibility/input_manifest.json').read_text())
    raw={item['path']:item['sha256'] for item in manifest['files'] if item['role']=='raw_input' and item['path'] in missing}
    ledgers={}
    for name,(stem,count) in LEDGERS.items():
        ledger=json.loads((ROOT/'results'/f'{stem}.json').read_text())
        assert not ledger.get('failures'),f'{name}: failed evaluations'
        # The original robustness schema predates the complete flag.
        if name != 'model_robustness':
            assert ledger['metadata'].get('complete'),f'{name}: incomplete run'
        assert len(ledger['evaluations'])==count,f'{name}: wrong evaluation count'
        verify_provenance(name,ledger,unavailable_inputs=raw)
        ledgers[name]=ledger
    headline=ledgers['selection_gap']
    assert headline['metadata']['n_unique_partition_digests']==6
    assert ledgers['model_robustness']['metadata']['n_unique_partition_digests']==6
    pairs=headline['partition_diagnostics']['same_fold_test_manufacturer_jaccards']
    assert len(pairs)==60 and all(0<=row['jaccard']<1 for row in pairs)
    print(f'PASS: {checked} input hashes; nine complete ledgers; available source provenance verified')
    for relative in missing:
        print(f'NOT CHECKED: licensed input absent: {relative}')
    print('Released headline means:')
    for field in ('diagnostic_r2','diagnostic_pairwise_accuracy'):
        values=[row[field] for row in headline['evaluations']]
        assert np.isfinite(values).all(),f'nonfinite {field}'
        print(f'  {field}: {np.mean(values):.6f}')
    print('These checks do not refit the models; use ./reproduce.sh for a full run.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
