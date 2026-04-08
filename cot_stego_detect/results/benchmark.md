# CoT stego detection benchmark

AUROC for each detector vs. each encoder.
Negative class = baseline batches; positive class = stego batches.

| detector | acrostic | length_parity | punctuation | synonym |
|---|---|---|---|---|
| chi_sq_template | 0.778 | 0.451 | 0.483 | 0.927 |
| punct_rate | 0.504 | 0.822 | 1.000 | 0.547 |
| length_parity | 0.457 | 0.731 | 1.000 | 0.529 |
| bigram_surprisal | 0.746 | 0.710 | 1.000 | 0.834 |
| ensemble | 0.810 | 0.824 | 1.000 | 0.948 |
