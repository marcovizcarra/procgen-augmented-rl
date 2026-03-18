# Stage-1 IQL Train+Eval Summary

| variant | train_mean | train_std | test_mean | test_std | gen_gap | ckpt |
| --- | --- | --- | --- | --- | --- | --- |
| shift4_noise_0p02 | 0.550 | 2.280 | 1.100 | 3.129 | 0.550 | runs/compare_bc_iql_L40_top3_seed1_steps20k_ep200/iql_shift4_noise_0p02/iql_ckpt.pt |
| shift4_scale_0p9_1p1 | 0.950 | 2.932 | 0.900 | 2.862 | -0.050 | runs/compare_bc_iql_L40_top3_seed1_steps20k_ep200/iql_shift4_scale_0p9_1p1/iql_ckpt.pt |
| shift4_scale_0p8_1p2 | 1.550 | 3.619 | 0.650 | 2.465 | -0.900 | runs/compare_bc_iql_L40_top3_seed1_steps20k_ep200/iql_shift4_scale_0p8_1p2/iql_ckpt.pt |
