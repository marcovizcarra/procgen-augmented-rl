# shift_pad4

- dataset_root: `data/stage1_datasets_L40`
- run_group: `stage1_datasets_L40`
- ckpt: `runs/stage1_datasets_L40/bc_shift_pad4/bc_ckpt.pt`

## Train
- steps: 20000
- batch_size: 256
- lr: 0.0003
- seed: 0
- elapsed_sec: 374.72
- skipped: False

## Eval
- episodes: 200
- split: train_start=0, train_levels=40, test_start=40, test_levels=500
- distribution_mode: hard
- train_mean/std: 2.100 / 4.073
- test_mean/std: 1.600 / 3.666
- gen_gap (test-train): -0.500
- eval_elapsed_sec: 92.72
