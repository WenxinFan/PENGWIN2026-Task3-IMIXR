# Clinical Val/Test Split

This split separates model selection from final local testing.

- `train`: 120 cases used by `trainer.fit`.
- `val`: 30 `val_select` cases, stratified by GT-only hardness among non-test cases. Use this for early stopping, checkpoint selection, alpha selection, and point-count selection.
- `test`: 20 fixed GT-hard cases. Do not use this for training decisions; run it only after a candidate is selected from `val`.

Manifest:

```text
configs/splits/clinical_val30_hard20_seed42.json
```

Fold name:

```text
val30_hard20_seed42
```

`val_select` cases:

```text
001 003 006 012 019 028 029 037 044 050
053 057 066 075 080 091 094 095 099 106
111 113 167 174 177 185 190 196 198 200
```

`hard_test` cases:

```text
199 086 031 058 103 153 178 156 197 008
025 052 079 011 101 024 092 005 108 073
```
