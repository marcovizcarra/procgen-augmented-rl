# Stage-1 IQL Train+Eval Summary

| variant | train_mean | train_std | test_mean | test_std | gen_gap | ckpt |
| --- | --- | --- | --- | --- | --- | --- |
| shift_pad8 | 0.950 | 2.932 | 1.350 | 3.417 | 0.400 | runs/iql_best_vs_bc_L40_rest8_seed0_steps20k/iql_shift_pad8/iql_ckpt.pt |
| shift_pad4 | 1.100 | 3.129 | 1.300 | 3.363 | 0.200 | runs/iql_best_vs_bc_L40_rest8_seed0_steps20k/iql_shift_pad4/iql_ckpt.pt |
| shift4_cutout24 | 1.200 | 3.250 | 1.150 | 3.190 | -0.050 | runs/iql_best_vs_bc_L40_rest8_seed0_steps20k/iql_shift4_cutout24/iql_ckpt.pt |
| shift4_cutout16 | 0.550 | 2.280 | 0.950 | 2.932 | 0.400 | runs/iql_best_vs_bc_L40_rest8_seed0_steps20k/iql_shift4_cutout16/iql_ckpt.pt |
| shift4_jitter_0p2 | 0.200 | 1.400 | 0.900 | 2.862 | 0.700 | runs/iql_best_vs_bc_L40_rest8_seed0_steps20k/iql_shift4_jitter_0p2/iql_ckpt.pt |
| shift4_paug_0p5 | 1.350 | 3.417 | 0.900 | 2.862 | -0.450 | runs/iql_best_vs_bc_L40_rest8_seed0_steps20k/iql_shift4_paug_0p5/iql_ckpt.pt |
| shift4_blur3x3 | 0.300 | 1.706 | 0.800 | 2.713 | 0.500 | runs/iql_best_vs_bc_L40_rest8_seed0_steps20k/iql_shift4_blur3x3/iql_ckpt.pt |
| shift4_jitter_0p4 | 0.400 | 1.960 | 0.600 | 2.375 | 0.200 | runs/iql_best_vs_bc_L40_rest8_seed0_steps20k/iql_shift4_jitter_0p4/iql_ckpt.pt |
