
The problems are used to evaluate the performance of the surrogate models. Now, we 
are implementing a set of problems, they are used for modeling, unconstrained optimization,
constrained optimization, and multi-objective optimization.

.. note::
    Till now, we only implement problems for **unconstrained single-objective optimization**, **unconstrained multi-fidelity single-objective optimization**, **constrained single-objective optimization**. 


Problems class structure
========================

Now, we want to introduce the basic :attr:`~mfpml.problems.function.Functions` class:

.. code-block:: python

   class Functions(ABC):
    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
      ...

    @staticmethod
    def f_der(x: np.ndarray) -> np.ndarray:
      ...
    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:
      ...

    @staticmethod
    def lf(x: np.ndarray, factor: float) -> np.ndarray:
      ...
    @staticmethod
    def hf_der(x: np.ndarray) -> np.ndarray:
      ...
    @staticmethod
    def lf_der(x: np.ndarray) -> np.ndarray:
      ...
    @staticmethod
    def f_cons(x: np.ndarray) -> np.ndarray:
      ...

    @staticmethod
    def hf_cons(x: np.ndarray) -> np.ndarray:
      ...

    @staticmethod
    def lf_cons(x: np.ndarray) -> np.ndarray:
      ...

Single fidelity problems
========================

Example for single fidelity function
------------------------------------

Here, an example of using function :attr:`~mfpml.problems.sf_functions` is given.
The *Forrester* function :attr:`~mfpml.problems.sf_functions.Forrester` is used to illustrate: 

.. code-block:: python

   import numpy as np
   from mfpml.problems.sf_functions import Forrester
   from mfpml.design_of_experiment.sf_samplers import LatinHyperCube

   # Initialize the single-fidelity function
   function = Forrester()
   print("Input domain:", function.input_domain)

   # Sample input space using Latin Hypercube Sampling
   sampler = LatinHyperCube(design_space=function.input_domain)
   sample_x = sampler.get_samples(num_samples=2, seed=1)
   print("Sampled inputs:", sample_x)

   # Evaluate the function
   sample_y = function.f(sample_x)
   print("Function outputs:", sample_y)


Implemented single fidelity Functions
-------------------------------------

======================== ========================================================================================
Methods                   API of sampling methods                                            
======================== ========================================================================================         
Forrester                  :attr:`~mfpml.problems.sf_functions.Forrester`
Branin                     :attr:`~mfpml.problems.sf_functions.Branin`
Hartmann3                  :attr:`~mfpml.problems.sf_functions.Hartmann3`
Hartmann6                  :attr:`~mfpml.problems.sf_functions.Hartmann6`
Sixhump                    :attr:`~mfpml.problems.sf_functions.Sixhump`
GoldPrice                  :attr:`~mfpml.problems.sf_functions.GoldPrice`
Sasena                     :attr:`~mfpml.problems.sf_functions.Sasena` 
Ackley                     :attr:`~mfpml.problems.sf_functions.Ackley`
AckleyN2                   :attr:`~mfpml.problems.sf_functions.AckleyN2`
Thevenot                   :attr:`~mfpml.problems.sf_functions.Thevenot`
======================== ========================================================================================


Multi fidelity problems
========================

Example for multi fidelity function
------------------------------------

Here, an example of using function :attr:`~mfpml.problems.mf_functions` is given.
The *Forrester_1a* function :attr:`~mfpml.problems.mf_functions.Forrester_1a.` is used to illustrate:


.. code-block:: python
   
   import numpy as np
   from mfpml.problems.mf_functions import Forrester_1b
   from mfpml.design_of_experiment.mf_samplers import MFSobolSequence

   # Initialize the multi-fidelity function
   function = Forrester_1b()
   design_space = function.input_domain
   print("Input domain:", design_space)

   # Sample input space using Multi-Fidelity Sobol Sequence
   sampler = MFSobolSequence(design_space=design_space, num_fidelity=2, nested=True)
   samples = sampler.get_samples(num_samples=[4, 11])
   print("Sampled inputs at two fidelities:", samples)

   # Evaluate the function
   sample_y = function(samples)
   print("Function outputs:", sample_y)


Implemented multi fidelity Functions
------------------------------------

================================    ========================================================================================
Methods                                API of sampling methods
================================    ========================================================================================
Forrester_1a                        :attr:`~mfpml.problems.mf_functions.Forrester_1a`
Forrester_1b                        :attr:`~mfpml.problems.mf_functions.Forrester_1b`
Forrester_1c                        :attr:`~mfpml.problems.mf_functions.Forrester_1c`
mf_Hartmann3                        :attr:`~mfpml.problems.mf_functions.mf_Hartmann3`
mf_Hartmann6                        :attr:`~mfpml.problems.mf_functions.mf_Hartmann6`
mf_Sixhump                          :attr:`~mfpml.problems.mf_functions.mf_Sixhump`
mf_Discontinuous                    :attr:`~mfpml.problems.mf_functions.mf_Discontinuous`
PhaseShiftedOscillations            :attr:`~mfpml.problems.mf_functions.PhaseShiftedOscillations`
ContinuousNonlinearCorrelation1D    :attr:`~mfpml.problems.mf_functions.ContinuousNonlinearCorrelation1D`
================================    ========================================================================================
