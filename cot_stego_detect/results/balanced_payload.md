# Balanced-payload evasion

SynonymEncoder, batch size 200, 40 paired runs.
AUROC for separating stego from clean batches.

| detector | unbalanced_5of7 | balanced_4of8 |
|---|---|---|
| chi_sq_template | 1.000 | 0.155 |
| punct_rate | 0.461 | 0.570 |
| length_parity | 0.503 | 0.521 |
| bigram_surprisal | 1.000 | 0.439 |
