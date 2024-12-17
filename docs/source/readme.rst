Summary
====================================================



The `mfpml` package is a toolkit for multi-fidelity probabilistic machine learning, Bayesian optimization based on single-/multi-fidelity data resources. This package is design to provide a platform of replicating existing single-/multi-fidelity Bayesian methods based on Gaussian process regression; and also a handy tool for developing new methods in the field of multi-fidelity probabilistic machine learning. Overall, this package has two main features:

1. **Basic methods**: fundamental implementations of popular methods for example Gaussian process regression, Co-Kriging and corresponding extensions \
2. **Advanced Methods**: advanced methods for Bayesian optimization, for instance, riable-fidelity optimization, active learning reliability analysis, multi-fidelity reliability analysis.


Basic Methods
-------------


The `mfpml` package provides essential tools and implementations across four key areas:

1. **Design of Experiments**
   
   - Provides efficient strategies for designing experiments to reduce cost and improve performance.

2. **Functions**
   
   - A collection of commonly used functions for experimental analysis and performance evaluation.

3. **Models**
   
   - **Kriging Model**:  
     A Gaussian process-based surrogate model for single-fidelity data.  
     :attr:`mfpml.models.gaussian_process`
   - **Multi-Fidelity Kriging Model**:  
     Extends Kriging to incorporate multi-fidelity data for better predictions.  
     :attr:`mfpml.models.mf_gaussian_process`

4. **Optimization**
   
   - **Evolutionary Algorithms**:  
     A metaheuristic approach for solving complex optimization problems.  
     :attr:`mfpml.optimization.evolutionary_algorithms`

   - **Single-Fidelity Bayesian Optimization (SFBO)**:  
     Bayesian optimization for expensive single-fidelity black-box functions.  
     :attr:`mfpml.optimization.sfbo`
   - **Multi-Fidelity Bayesian Optimization (MFBO)**:  
     An advanced Bayesian optimization framework that leverages multi-fidelity data.  
     :attr:`mfpml.optimization.mfbo`


Advanced methods
----------------

The `mfpml` package also includes advanced methods, implemented based on the following publications. These methods push the boundaries of multi-fidelity modeling, optimization, and reliability analysis.

1. **Multi-Fidelity Lower Confidence Bounding**  
   *Jiang, Ping, et al.*  
   "Variable-fidelity lower confidence bounding approach for engineering optimization problems with expensive simulations."  
   *AIAA Journal*, 57(12), 2019, pp. 5416-5430.  

   - **Focus**: Multi-fidelity optimization for computationally expensive problems.

2. **Enhanced Multi-Fidelity Optimization**  
   *Cheng, Ji, Qiao Lin, and Jiaxiang Yi.*  
   "An enhanced variable-fidelity optimization approach for constrained optimization problems and its parallelization."  
   *Structural and Multidisciplinary Optimization*, 65(7), 2022, p. 188.  

   - **Focus**: Improved constrained optimization and parallel implementation.

3. **Adaptive Kriging-Based Reliability Analysis**  
   *Yi, Jiaxiang, et al.*  
   "Efficient adaptive Kriging-based reliability analysis combining new learning function and error-based stopping criterion."  
   *Structural and Multidisciplinary Optimization*, 62, 2020, pp. 2517-2536.  

   - **Focus**: Enhanced Kriging reliability analysis with adaptive learning.

4. **Active-Learning Multi-Fidelity Kriging for Reliability Analysis**  
   *Yi, Jiaxiang, et al.*  
   "An active-learning method based on multi-fidelity Kriging model for structural reliability analysis."  
   *Structural and Multidisciplinary Optimization*, 63, 2021, pp. 173-195.  
   
   - **Focus**: Combining active learning with multi-fidelity Kriging for reliability analysis.

5. **Fidelity Selection Strategy for Multi-Fidelity Reliability Analysis**  
   *Yi, Jiaxiang, Yuansheng Cheng, and Jun Liu.*  
   "A novel fidelity selection strategy-guided multifidelity Kriging algorithm for structural reliability analysis."  
   *Reliability Engineering & System Safety*, 219, 2022, p. 108247. 

   - **Focus**: Novel fidelity selection strategies to enhance reliability analysis.


Authorship
----------

**Authors**:

- **Jiaxiang Yi**\ [1]_ | `yagafighting@gmail.com <mailto:yagafighting@gmail.com>`_  

- **Ji Cheng**\ [2]_ | `jicheng9617@gmail.com <mailto:jicheng9617@gmail.com>`_  

**Affiliations**:  

.. [1] Delft University of Technology  
.. [2] City University of Hong Kong  

Community Support
-----------------

If you encounter any issues, bugs, or problems with this package, please:  

- **Raise an issue** on the official GitHub page
- **Contact the authors** via email

Your feedback is highly valued and will help improve the package!


License
-------

Copyright © 2024, **Jiaxiang Yi** and **Ji Cheng**  
All rights reserved.

This project is licensed under the **MIT License**. For details, refer to the `LICENSE` file included with the repository.


