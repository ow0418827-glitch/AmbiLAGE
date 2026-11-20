# Stage 2 v3 干净原图基准校准（Gate 3）

| 编码器 | 评测集 | Clean T1 | Clean T3 | Clean T5 | Clean 均名 | 子家族 T5 (desc / para / search) |
|---|---|---:|---:|---:|---:|---:|
| CLIP-ViT-L/14 | `training_oracle_v3` | 13.3% | 25.0% | 35.0% | 13.08 | 33.3% / 33.3% / 41.7% |
| CLIP-ViT-L/14 | `overlap_30_v3` | 11.1% | 21.7% | 34.4% | 13.77 | 32.2% / 29.6% / 47.2% |
| CLIP-ViT-L/14 | `zero_overlap_v3` | 9.4% | 20.6% | 32.2% | 13.71 | 33.3% / 24.1% / 41.7% |
| OpenCLIP-bigG-14 | `training_oracle_v3` | 14.4% | 29.4% | 42.8% | 9.34 | 42.2% / 37.0% / 52.8% |
| OpenCLIP-bigG-14 | `overlap_30_v3` | 11.7% | 24.4% | 36.7% | 10.78 | 36.7% / 31.5% / 44.4% |
| OpenCLIP-bigG-14 | `zero_overlap_v3` | 11.1% | 22.8% | 34.4% | 10.59 | 35.6% / 31.5% / 36.1% |
| SigLIP-SO400M | `training_oracle_v3` | 13.3% | 30.0% | 46.7% | 9.23 | 51.1% / 37.0% / 50.0% |
| SigLIP-SO400M | `overlap_30_v3` | 10.6% | 24.4% | 37.8% | 10.05 | 40.0% / 33.3% / 38.9% |
| SigLIP-SO400M | `zero_overlap_v3` | 10.0% | 23.9% | 32.2% | 10.17 | 33.3% / 31.5% / 30.6% |

- Gate 3 (clean zero_overlap_v3 <= oracle_v3 + 1.0 pp): **PASSED**
