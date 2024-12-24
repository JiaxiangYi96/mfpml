<p align="center">
    <img src="docs/source/figures/logo.png" width="70%" align="center">
</p>

---

[**Documentation**](https://jiaxiangyi96.github.io/mfpml/)
| [**Installation**](https://jiaxiangyi96.github.io/mfpml/get_started.html)
| [**GitHub**](https://github.com/JiaxiangYi96/mfpml)
| [**Tutorials**](https://github.com/JiaxiangYi96/mfpml/tree/main/tutorials)


# MFPML: Multi-Fidelity Probabilistic Machine Learning Library

**MFPML** is a Python library for multi-/single-fidelity probabilistic machine learning. It provides tools for **design of experiments**, **surrogate modeling**, **Bayesian optimization**, and **visualization** – empowering researchers and engineers to optimize expensive, black-box systems efficiently.
 
---

## 👥 **Authorship**

**Authors**:  
- Jiaxiang Yi (yagafighting@gmail.com) [1]  
- Ji Cheng (jicheng9617@gmail.com) [2]  

**Affiliations**:  
- [1] Delft University of Technology  
- [2] City University of Hong Kong  

## 🚀 **Key Features**

### 1. Multi-Fidelity Design of Experiments (DoE)
Efficient sampling methods to generate design points for optimization and modeling:
- **Multi-Fidelity Samplers**: Sampling with **Sobol sequences** and **Latin Hypercube Sampling (LHS)**.
- **Single-Fidelity Samplers**: Random, Sobol, and LHS sampling.

Example:
```python
import numpy as np
from mfpml.design_of_experiment.mf_samplers import MFSobolSequence
# Define the input design space
design_space = np.array([[0, 1], [0, 1]]) 
# define the sampler
sampler = MFSobolSequence(design_space=design_space, num_fidelity=2, nested=True)
# get samples 
samples = sampler.get_samples(num_samples=[2, 5])
# print samples
print(samples)
    [array([[0.64763958, 0.28450888],
            [0.36683413, 0.68135563]]), 
     array([[0.64763958, 0.28450888],
            [0.36683413, 0.68135563],
            [0.48028161, 0.47841079],
            [0.51099956, 0.61233338],
            [0.92215368, 0.08658168]])]

```

---

### 2. Advanced Surrogate Models
Model expensive functions with state-of-the-art surrogate techniques:
- **Gaussian Process Regression (GPR)**
- **Co-Kriging** for multi-fidelity modeling
- **Hierarchical and Scaled Kriging** variants

[**Example**](https://jiaxiangyi96.github.io/mfpml/multi_fidelity_kriging_models.html):
```python
from mfpml.models.co_kriging import CoKriging

model = CoKriging(design_space=input_domain, # np.ndarray
                optimizer_restart=10)
model.train(X_train, Y_train)  # Train the model
Y_pred, sigma = model.predict(X_test, return_std=True)  # Predict outputs
```

---

### 3. Multi-Fidelity Bayesian Optimization
Efficient optimization algorithms to solve black-box problems:
- **Single-Fidelity Bayesian Optimization, see [Example](https://jiaxiangyi96.github.io/mfpml/optimization.html#single-fidelity-bayesian-optimization)**
- **Multi-Fidelity Bayesian Optimization, see [Example](https://jiaxiangyi96.github.io/mfpml/optimization.html#multi-fidelity-bayesian-optimization)**
```python
from mfpml.problems.mf_functions import Forrester_1a
from mfpml.optimization.mf_uncons_bo import mfUnConsBayesOpt
from mfpml.optimization.mf_acqusitions import  AugmentedEI,


# define problem
problem = Forrester_1a()
# define the optimizer
optimizer = mfUnConsBayesOpt(problem=problem,
                            acquisition=VFEI(),
                            num_init=[5, 20],
                            )
# execute the optimizer
optimizer.run_optimizer(max_iter=20, stopping_error=0.01, cost_ratio=5.0)
```
---

## 📦 **Installation**

Install MFPML via `pip`:
```bash
pip install mfpml
```

Or install from source:
```bash
git clone https://github.com/your_username/mfpml.git
cd mfpml
pip install .
```

---

## 📚 **Documentation**

You can also explore the tutorials provided by sphinx online [documentation](https://jiaxiangyi96.github.io/mfpml/index.html) or by Jupyter [Notebooks](https://github.com/JiaxiangYi96/mfpml/tree/main/tutorials):


---

## ✅ **Testing**

To ensure everything works correctly, run the test suite using `pytest`:
```bash
pytest tests/
```

---

## 🤝 **Contributing**

Contributions are welcome! Follow these steps to contribute:
1. Fork the repository.
2. Create a new feature branch: `git checkout -b feature/your-feature-name`.
3. Commit your changes: `git commit -m "Add new feature"`.
4. Push to your branch: `git push origin feature/your-feature-name`.
5. Submit a pull request.

---

## 🌟 **Why Use MFPML?**

- **Efficiency**: Optimize expensive simulations with fewer samples.
- **Flexibility**: Use single or multi-fidelity methods tailored to your needs.
- **Integration**: Easy to integrate into engineering and scientific workflows.
- **Open Source**: Fully customizable and extendable.

---


## 📄 **License**

This project is licensed under the MIT License.

---

## 🙌 **Acknowledgments**

We appreciate the open-source community and contributors who make this project better.

![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

---
## 📝 **References**

1. Jiang, Ping, et al. "Variable-fidelity lower confidence bounding approach for engineering optimization problems with expensive simulations." *AIAA Journal* 57.12 (2019): 5416-5430.  
2. Cheng, Ji, Qiao Lin, and Jiaxiang Yi. "An enhanced variable-fidelity optimization approach for constrained optimization problems and its parallelization." *Structural and Multidisciplinary Optimization* 65.7 (2022): 188.  
3. Yi, Jiaxiang, et al. "Efficient adaptive Kriging-based reliability analysis combining new learning function and error-based stopping criterion." *Structural and Multidisciplinary Optimization* 62 (2020): 2517-2536.  
4. Yi, Jiaxiang, et al. "An active-learning method based on multi-fidelity Kriging model for structural reliability analysis." *Structural and Multidisciplinary Optimization* 63 (2021): 173-195.  
5. Yi, Jiaxiang, Yuansheng Cheng, and Jun Liu. "A novel fidelity selection strategy-guided multifidelity kriging algorithm for structural reliability analysis." *Reliability Engineering & System Safety* 219 (2022): 108247. 