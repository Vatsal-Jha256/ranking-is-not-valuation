#!/usr/bin/env bash
# Verify inputs, refit the nine experiments, compare results, and export artifacts.
# Run through ./reproduce.sh to use the pinned environment.
set -euo pipefail

workspace="/workspace"
artifacts="${ARTIFACT_DIR:-/artifacts}"
results="${workspace}/results"
figures="${workspace}/figures"

# Reported experiments, as
# "<ledger name>:<runner path>", in dependency order.
experiments=(
  "exp_submission_harmonized_v4:experiments/selection_gap/run_selection_gap.py"
  "exp_model_robustness_v3:experiments/model_robustness/run_model_families.py"
  "exp_calibration_structure_v1:experiments/calibration_premise/run_premise_audit.py"
  "exp_uncertainty_policies_v1:experiments/reporting_policies/run_policy_comparison.py"
  "exp_candidate_sensitivity_v1:experiments/pool_sensitivity/run_pool_sweep.py"
  "exp_temporal_holdout_v1:experiments/temporal_holdout/run_temporal_holdout.py"
  "exp_beta_shrinkage_v2:experiments/supplementary/run_beta_shrinkage.py"
  "exp_knapsack_v2:experiments/supplementary/run_budget_allocation.py"
  "exp_stratified_quantile_v1:experiments/reporting_policies/run_stratified_quantile.py"
)
# The first two ledgers keep dedicated verifier flags; the other seven are compared
# by name.
review=(
  "calibration_structure:exp_calibration_structure_v1"
  "uncertainty_policies:exp_uncertainty_policies_v1"
  "candidate_sensitivity:exp_candidate_sensitivity_v1"
  "temporal_holdout:exp_temporal_holdout_v1"
  "beta_shrinkage:exp_beta_shrinkage_v2"
  "budget_allocation:exp_knapsack_v2"
  "stratified_quantile:exp_stratified_quantile_v1"
)
appendix_figures=(
  figA1_premise_audit
  figA2_selection_vs_reporting
  figA3_pool_sensitivity
  figA4_temporal_holdout
  figA5_accepted_policies
  figA6_supplementary_checks
)
mkdir -p "${artifacts}" /tmp/matplotlib /tmp/locked

python "${workspace}/reproducibility/generate_manifest.py" verify

for entry in "${experiments[@]}"; do
  cp "${results}/${entry%%:*}.json" "/tmp/locked/${entry%%:*}.json"
done

for entry in "${experiments[@]}"; do
  python "${workspace}/${entry##*:}"
done

python "${workspace}/figures/gen_submission_figure.py"
python "${workspace}/figures/gen_appendix_figures.py"

verify_args=(
  --reference "/tmp/locked/exp_submission_harmonized_v4.json"
  --candidate "${results}/exp_submission_harmonized_v4.json"
  --robustness-reference "/tmp/locked/exp_model_robustness_v3.json"
  --robustness-candidate "${results}/exp_model_robustness_v3.json"
  --figure "${figures}/fig1_submission_story.pdf"
  --report "${artifacts}/verification.json"
)
for entry in "${review[@]}"; do
  verify_args+=(--review-pair "${entry%%:*}" "/tmp/locked/${entry##*:}.json" \
                "${results}/${entry##*:}.json")
done
for name in "${appendix_figures[@]}"; do
  verify_args+=(--appendix-figure "${figures}/${name}.pdf")
done
python -m pip freeze --all | LC_ALL=C sort > "${artifacts}/python-environment.txt"
python "${workspace}/reproducibility/verify_outputs.py" "${verify_args[@]}"

for entry in "${experiments[@]}"; do
  cp "${results}/${entry%%:*}.json" "${artifacts}/${entry%%:*}.reproduced.json"
done
cp "${figures}/fig1_submission_story.pdf" "${artifacts}/fig1_submission_story.reproduced.pdf"
cp "${figures}/specs/fig1_rank_then_report.json" "${artifacts}/fig1_rank_then_report.spec.json"
cp "${figures}/fig1_rank_then_report.svg" "${artifacts}/fig1_rank_then_report.svg"
cp "${figures}/fig1_rank_then_report.pdf" "${artifacts}/fig1_rank_then_report.pdf"
for name in "${appendix_figures[@]}"; do
  cp "${figures}/${name}.pdf" "${artifacts}/${name}.reproduced.pdf"
done
cp "${workspace}/reproducibility/input_manifest.json" "${artifacts}/input_manifest.json"

cd "${artifacts}"
sha256sum \
  *.reproduced.json \
  *.reproduced.pdf \
  fig1_rank_then_report.spec.json \
  fig1_rank_then_report.svg \
  fig1_rank_then_report.pdf \
  input_manifest.json \
  python-environment.txt \
  verification.json > SHA256SUMS

echo "PASS: locked experiments, figures, provenance, and artifact hashes verified"
