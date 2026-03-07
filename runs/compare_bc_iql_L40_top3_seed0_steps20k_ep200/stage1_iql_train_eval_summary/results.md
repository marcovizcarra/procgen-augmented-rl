# Stage-1 IQL Train+Eval Summary

| variant | train_mean | train_std | test_mean | test_std | gen_gap | ckpt |
| --- | --- | --- | --- | --- | --- | --- |
| shift4_noise_0p02 | 0.750 | 2.634 | 1.600 | 3.666 | 0.850 | runs/compare_bc_iql_L40_top3_seed0_steps20k_ep200/iql_shift4_noise_0p02/iql_ckpt.pt |
| shift4_scale_0p9_1p1 | 1.650 | 3.712 | 1.150 | 3.190 | -0.500 | runs/compare_bc_iql_L40_top3_seed0_steps20k_ep200/iql_shift4_scale_0p9_1p1/iql_ckpt.pt |
| shift4_scale_0p8_1p2 | 0.750 | 2.634 | 0.900 | 2.862 | 0.150 | runs/compare_bc_iql_L40_top3_seed0_steps20k_ep200/iql_shift4_scale_0p8_1p2/iql_ckpt.pt |
