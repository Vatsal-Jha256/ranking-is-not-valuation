# E5. Stratified empirical-quantile reporter (reviewer xqki's simpler baseline)

Written before the run, 2026-10-12.

Question. Does the learned heteroscedastic conformal floor add anything over a reporter that
uses only strength-class empirical quantiles and no learned scale model?

Design. Same 24 manufacturer-disjoint evaluations, same candidate sets of ten, same RNG
streams and argmin selection as E2, so the selected product and its realized advantage are
identical to `conformal_report` by construction. Two reporters replace the number attached to
the argmin winner W, each calibrated only on the calibration block:

- `class_shift_report`: reported log GWP = alpha + beta X_W + q_c, where q_c is the
  finite-sample-corrected 90% quantile of the winner's residual T - (alpha + beta X) over 500
  calibration candidate sets of the same size drawn within strength class c. One scalar per
  class, no scale model.
- `class_lookup_report`: reported log GWP = u_c, the finite-sample-corrected 90% quantile of
  declared log GWP among calibration products of class c. No model at all.

Classes with fewer than 30 calibration rows are not scored (same eligibility as E2).

Metrics, identical to E2: mean signed gap, share of cases overstated above 0.05 log units,
exceedance of the reported value (gap > 0), CVaR90, and share of cases whose implied saving
floor against the class-median incumbent is positive.

Prespecified reading. The learned reporter is justified only if it beats `class_shift_report`
on severe overstatement or positive-floor rate by more than the split/fold SD. If not, the
manuscript says the learned scale model is not justified by this corpus. No threshold is
tuned on test data. Failure conditions: any non-finite metric or fewer than 24 completed
evaluations.
