# Attribution

This repository contains code only. The assets it audits are fetched at pinned revisions by `make download` and are not redistributed here.

| Dataset | Licence | Pinned revision |
|---|---|---|
| [projectsim/bottles-and-containers](https://huggingface.co/datasets/projectsim/bottles-and-containers) | CC BY 4.0 | `2d5aa958dd926b890cdf3436f026c2d27b6d695b` |
| [projectsim/yam-real-to-sim-assets](https://huggingface.co/datasets/projectsim/yam-real-to-sim-assets) | CC BY 4.0 | `bc56a211d00914160ef6297935598d8e8f16dc2f` |
| [projectsim/yam-real-to-sim-assets-default](https://huggingface.co/datasets/projectsim/yam-real-to-sim-assets-default) | CC BY 4.0 | `8bdfdedf9834a407f2e5db26a8c77abd685b4828` |

Asset data © Kaedim / projectsim. The two real-to-sim arms are described in Verma, Cilio, Christodoulou, Fermelis, Ng, Chung and Liang, *Measuring Asset and Scene Reconstruction Effects in Real-to-Sim Robot Evaluation*, Kaedim, 2026 ([arXiv:2610.00731](https://arxiv.org/abs/2610.00731)). `configs/paper_results.json` holds the per-cell results table from [Kaedim/yam-sim-harness-oss](https://github.com/Kaedim/yam-sim-harness-oss/blob/c44079e895c3bd4cec854ccdb4533180edc632fb/docs/RESULTS.md), parsed at that commit.

Real-world references used by check C3:

- International Aluminium Institute, "Lightweight" (alustory.international-aluminium.org/packaging/lightweight): a 330 ml can weighs about 13 g; can walls can be produced at 0.097 mm.
- Ardagh Metal Packaging, 330 ml sleek can specification 04 5 1 58 330 1 02: nominal finished can weight 9.6 ± 0.6 g.
