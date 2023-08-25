# mfpml

This repository aims to provide a package for multi-fidelity probabilistic machine learning. The package is developed by Jiaxiang Yi and Ji Cheng based on their learning curve on multi-fidelity probabilistic macgin learning, and multi-fidelity Bayesian optimization, and multi-fidelity reliability analysis.

Overall, this mfpml package has two main goals, the first one is to provide basic code on implement typical methods in modeling, optimization, and reliability analysis field. Then, based on the basic code, we also provide some advanced methods based on our publications.

---

(1) Basic methods

**Models**

- Kriging model `mfpml.models.sf_gpr`
- Multi-fidelity Kriging model `mfpml.models.mf_gprs`

**Optimizations**

- Evolutionary algorithms `mfpml.optimization.evolutionary_algorithms`

- Single fidelity Bayesian optimization `mfpml.optimization.sfbo`
- Multi-fidelity Bayesian optimization `mfpml.optimization.mfbo`

**Reliability analysis**

- Active learning reliability analysis
- Multi-fidelity reliability analysis

(2) Advanced methods

For the advanced methods, we will provide code based on our publications.
please check out those papers:

- Jiang, Ping, et al. "Variable-fidelity lower confidence bounding approach for engineering optimization problems with expensive simulations." AIAA Journal 57.12 (2019): 5416-5430.
- Cheng, Ji, Qiao Lin, and Jiaxiang Yi. "An enhanced variable-fidelity optimization approach for constrained optimization problems and its parallelization." Structural and Multidisciplinary Optimization 65.7 (2022): 188.
- Yi, Jiaxiang, et al. "Efficient adaptive Kriging-based reliability analysis combining new learning function and error-based stopping criterion." Structural and Multidisciplinary Optimization 62 (2020): 2517-2536.
- Yi, Jiaxiang, et al. "An active-learning method based on multi-fidelity Kriging model for structural reliability analysis." Structural and Multidisciplinary Optimization 63 (2021): 173-195.
- Yi, Jiaxiang, Yuansheng Cheng, and Jun Liu. "A novel fidelity selection strategy-guided multifidelity kriging algorithm for structural reliability analysis." Reliability Engineering & System Safety 219 (2022): 108247.

## Authorship

**Authors**:

- Jiaxiang Yi(yagafighting@gmail.com)[1]
- Ji Cheng (jicheng9617@gmail.com)[2]

**Authors affiliation:**

- [1] Delft University of Technology

- [2] City University of Hong Kong

## Community Support

If you find any issues, bugs or problems with this package, you can raise an issue
on the github page, or contact the Authors directly.

## License

Copyright 2023, Jiaxiang Yi and Ji Cheng

All rights reserved.
