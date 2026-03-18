# Stage-1 IQL Train+Eval Summary

| variant | train_mean | train_std | test_mean | test_std | gen_gap | ckpt |
| --- | --- | --- | --- | --- | --- | --- |
| shift4_blur3x3 | 2.600 | 4.386 | 2.300 | 4.208 | -0.300 | runs/iql_stepsweep_L40_steps6000_seed0_steps6000_seed0/iql_shift4_blur3x3/iql_ckpt.pt |
| shift4_paug_0p5 | 1.700 | 3.756 | 2.100 | 4.073 | 0.400 | runs/iql_stepsweep_L40_steps6000_seed0_steps6000_seed0/iql_shift4_paug_0p5/iql_ckpt.pt |
| shift4_jitter_0p2 | 1.500 | 3.571 | 1.850 | 3.883 | 0.350 | runs/iql_stepsweep_L40_steps6000_seed0_steps6000_seed0/iql_shift4_jitter_0p2/iql_ckpt.pt |
| shift4_scale_0p9_1p1 | 1.200 | 3.250 | 1.800 | 3.842 | 0.600 | runs/iql_stepsweep_L40_steps6000_seed0_steps6000_seed0/iql_shift4_scale_0p9_1p1/iql_ckpt.pt |
| shift_pad8 | 2.250 | 4.176 | 1.700 | 3.756 | -0.550 | runs/iql_stepsweep_L40_steps6000_seed0_steps6000_seed0/iql_shift_pad8/iql_ckpt.pt |
| shift4_noise_0p02 | 1.300 | 3.363 | 1.600 | 3.666 | 0.300 | runs/iql_stepsweep_L40_steps6000_seed0_steps6000_seed0/iql_shift4_noise_0p02/iql_ckpt.pt |
| baseline_none | 1.350 | 3.417 | 1.550 | 3.619 | 0.200 | runs/iql_stepsweep_L40_steps6000_seed0_steps6000_seed0/iql_baseline_none/iql_ckpt.pt |
| shift4_jitter_0p4 | 1.600 | 3.666 | 1.500 | 3.571 | -0.100 | runs/iql_stepsweep_L40_steps6000_seed0_steps6000_seed0/iql_shift4_jitter_0p4/iql_ckpt.pt |
| shift_pad4 | 1.400 | 3.470 | 1.250 | 3.307 | -0.150 | runs/iql_stepsweep_L40_steps6000_seed0_steps6000_seed0/iql_shift_pad4/iql_ckpt.pt |
| shift4_scale_0p8_1p2 | 1.200 | 3.250 | 0.900 | 2.862 | -0.300 | runs/iql_stepsweep_L40_steps6000_seed0_steps6000_seed0/iql_shift4_scale_0p8_1p2/iql_ckpt.pt |
| shift4_cutout16 | 0.650 | 2.465 | 0.550 | 2.280 | -0.100 | runs/iql_stepsweep_L40_steps6000_seed0_steps6000_seed0/iql_shift4_cutout16/iql_ckpt.pt |
| shift4_cutout24 | 0.550 | 2.280 | 0.550 | 2.280 | 0.000 | runs/iql_stepsweep_L40_steps6000_seed0_steps6000_seed0/iql_shift4_cutout24/iql_ckpt.pt |
