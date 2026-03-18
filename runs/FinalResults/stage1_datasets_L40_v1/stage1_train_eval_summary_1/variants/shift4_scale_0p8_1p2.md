# shift4_scale_0p8_1p2

- dataset_root: `data/stage1_datasets_L40`
- run_group: `stage1_datasets_L40`
- ckpt: `runs/stage1_datasets_L40/bc_shift4_scale_0p8_1p2/bc_ckpt.pt`

## Train
- steps: 20000
- batch_size: 256
- lr: 0.0003
- seed: 0
- elapsed_sec: 458.20
- skipped: False

## Eval
- episodes: 200
- split: train_start=0, train_levels=40, test_start=40, test_levels=500
- distribution_mode: hard
- train_mean/std: 3.350 / 4.720
- test_mean/std: 2.350 / 4.240
- gen_gap (test-train): -1.000
- eval_elapsed_sec: 77.80
