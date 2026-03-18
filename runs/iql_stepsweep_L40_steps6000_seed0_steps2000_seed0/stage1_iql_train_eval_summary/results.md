# Stage-1 IQL Train+Eval Summary

| variant | train_mean | train_std | test_mean | test_std | gen_gap | ckpt |
| --- | --- | --- | --- | --- | --- | --- |
| shift4_scale_0p9_1p1 | 2.650 | 4.413 | 3.050 | 4.604 | 0.400 | runs/iql_stepsweep_L40_steps6000_seed0_steps2000_seed0/iql_shift4_scale_0p9_1p1/iql_ckpt.pt |
| shift4_jitter_0p4 | 3.050 | 4.604 | 2.600 | 4.386 | -0.450 | runs/iql_stepsweep_L40_steps6000_seed0_steps2000_seed0/iql_shift4_jitter_0p4/iql_ckpt.pt |
| shift4_blur3x3 | 2.150 | 4.108 | 2.250 | 4.176 | 0.100 | runs/iql_stepsweep_L40_steps6000_seed0_steps2000_seed0/iql_shift4_blur3x3/iql_ckpt.pt |
| shift4_jitter_0p2 | 2.150 | 4.108 | 2.250 | 4.176 | 0.100 | runs/iql_stepsweep_L40_steps6000_seed0_steps2000_seed0/iql_shift4_jitter_0p2/iql_ckpt.pt |
| shift4_paug_0p5 | 3.050 | 4.604 | 2.250 | 4.176 | -0.800 | runs/iql_stepsweep_L40_steps6000_seed0_steps2000_seed0/iql_shift4_paug_0p5/iql_ckpt.pt |
| shift4_noise_0p02 | 3.100 | 4.625 | 2.200 | 4.142 | -0.900 | runs/iql_stepsweep_L40_steps6000_seed0_steps2000_seed0/iql_shift4_noise_0p02/iql_ckpt.pt |
| shift4_scale_0p8_1p2 | 1.950 | 3.962 | 2.050 | 4.037 | 0.100 | runs/iql_stepsweep_L40_steps6000_seed0_steps2000_seed0/iql_shift4_scale_0p8_1p2/iql_ckpt.pt |
| baseline_none | 1.700 | 3.756 | 2.000 | 4.000 | 0.300 | runs/iql_stepsweep_L40_steps6000_seed0_steps2000_seed0/iql_baseline_none/iql_ckpt.pt |
| shift4_cutout16 | 1.050 | 3.066 | 2.000 | 4.000 | 0.950 | runs/iql_stepsweep_L40_steps6000_seed0_steps2000_seed0/iql_shift4_cutout16/iql_ckpt.pt |
| shift_pad4 | 2.750 | 4.465 | 1.900 | 3.923 | -0.850 | runs/iql_stepsweep_L40_steps6000_seed0_steps2000_seed0/iql_shift_pad4/iql_ckpt.pt |
| shift_pad8 | 2.150 | 4.108 | 1.900 | 3.923 | -0.250 | runs/iql_stepsweep_L40_steps6000_seed0_steps2000_seed0/iql_shift_pad8/iql_ckpt.pt |
| shift4_cutout24 | 2.450 | 4.301 | 1.650 | 3.712 | -0.800 | runs/iql_stepsweep_L40_steps6000_seed0_steps2000_seed0/iql_shift4_cutout24/iql_ckpt.pt |
