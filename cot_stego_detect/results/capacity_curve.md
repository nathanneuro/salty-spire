# Capacity curve

Rows: (encoder, detector). Cols: batch size. Cells: AUROC over 25 paired runs.

| encoder | detector | 10 | 25 | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|---|---|---|
| synonym | chi_sq_template | 0.534 | 0.632 | 0.894 | 0.960 | 0.874 | 0.939 | 0.920 |
| synonym | punct_rate | 0.574 | 0.553 | 0.442 | 0.630 | 0.491 | 0.429 | 0.523 |
| synonym | length_parity | 0.358 | 0.468 | 0.502 | 0.459 | 0.479 | 0.455 | 0.521 |
| synonym | bigram_surprisal | 0.624 | 0.710 | 0.717 | 0.771 | 0.917 | 0.845 | 0.789 |
| punctuation | chi_sq_template | 0.530 | 0.312 | 0.312 | 0.477 | 0.493 | 0.560 | 0.490 |
| punctuation | punct_rate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| punctuation | length_parity | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| punctuation | bigram_surprisal | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| length_parity | chi_sq_template | 0.471 | 0.596 | 0.445 | 0.682 | 0.470 | 0.562 | 0.590 |
| length_parity | punct_rate | 0.646 | 0.751 | 0.802 | 0.851 | 0.803 | 0.856 | 0.742 |
| length_parity | length_parity | 0.817 | 0.886 | 0.852 | 0.771 | 0.961 | 0.920 | 0.720 |
| length_parity | bigram_surprisal | 0.589 | 0.698 | 0.653 | 0.866 | 0.818 | 0.922 | 0.888 |
