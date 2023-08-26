Problems 
===========

The problems are used to evaluate the performance of the surrogate models. Now, we 
are implementing a set of problems, they are used for modeling, unconstrained optimization,
constrained optimization, and multi-objective optimization.

.. note::
   Till now, we only implement the problems for unconstrained optimization for 
   single fidelity and multi fidelity single objective optimization.

Problems class structure
------------------------

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

Here, an example of using function :attr:`~mfpml.problems.single_fidelity_functions.SingleFidelityFunctions` is given.
The *Forrester* function :attr:`~mfpml.problems.single_fidelity_functions.Forrester.` is used to illustrate: 

.. code-block:: python

   from mfpml.problems.single_fidelity_functions import Forrester
   forrester = Forrester()
   # output the design space
   forrester.design_space
   >>> 
   {'x': [0.0, 1.0]}
   # output the input domain
   forrester.input_domain
   >>> 
   [[0. 1.]]
   # evaluate the function
   x = np.array([0.5])
   y = forrester.f(x)
   >>> [[0.90929743]]


Implemented single fidelity Functions
-------------------------------------

======================== ========================================================================================
Methods                   API of sampling methods                                            
======================== ========================================================================================         
Forrester                  :attr:`~mfpml.problems.single_fidelity_functions.Forrester`
Branin                     :attr:`~mfpml.problems.single_fidelity_functions.Branin`
Hartmann3                  :attr:`~mfpml.problems.single_fidelity_functions.Hartmann3`
Hartmann6                  :attr:`~mfpml.problems.single_fidelity_functions.Hartmann6`
Sixhump                    :attr:`~mfpml.problems.single_fidelity_functions.Sixhump`
GoldPrice                  :attr:`~mfpml.problems.single_fidelity_functions.GoldPrice`
Sasena                     :attr:`~mfpml.problems.single_fidelity_functions.Sasena` 
Ackley                     :attr:`~mfpml.problems.single_fidelity_functions.Ackley`
AckleyN2                   :attr:`~mfpml.problems.single_fidelity_functions.AckleyN2`
Thevenot                   :attr:`~mfpml.problems.single_fidelity_functions.Thevenot`
======================== ========================================================================================


Multi fidelity problems
========================

Example for multi fidelity function
------------------------------------

Here, an example of using function :attr:`~mfpml.problems.multi_fidelity_functions.MultiFidelityFunctions` is given.
The *Forrester_1a* function :attr:`~mfpml.problems.multi_fidelity_functions.Forrester_1a.` is used to illustrate:


.. code-block:: python
   
      from mfpml.problems.multi_fidelity_functions import Forrester_1a
      forrester_1a = Forrester_1a()
      # output the design space
      forrester_1a.design_space
      >>> 
      {'x': [0.0, 1.0]}
      # output the input domain
      forrester_1a.input_domain
      >>> 
      [[0. 1.]]
      # evaluate the function
      x = {'hf': np.array([[0.80792223]]),
                  'lf': np.array([[0.80792223]])}
      y["hf"] = function.hf(x["hf"])
      y["lf"] = function.lf(x["lf"])
      >>>
      {'hf': array([[-4.49853898]]), 'lf': array([[-4.17004719]])}


Implemented multi fidelity Functions
------------------------------------

================================    ========================================================================================
Methods                                API of sampling methods
================================    ========================================================================================
Forrester_1a                        :attr:`~mfpml.problems.multi_fidelity_functions.Forrester_1a`
Forrester_1b                        :attr:`~mfpml.problems.multi_fidelity_functions.Forrester_1b`
Forrester_1c                        :attr:`~mfpml.problems.multi_fidelity_functions.Forrester_1c`
mf_Hartmann3                        :attr:`~mfpml.problems.multi_fidelity_functions.mf_Hartmann3`
mf_Hartmann6                        :attr:`~mfpml.problems.multi_fidelity_functions.mf_Hartmann6`
mf_Sixhump                          :attr:`~mfpml.problems.multi_fidelity_functions.mf_Sixhump`
mf_Discontinuous                    :attr:`~mfpml.problems.multi_fidelity_functions.mf_Discontinuous`
PhaseShiftedOscillations            :attr:`~mfpml.problems.multi_fidelity_functions.PhaseShiftedOscillations`
ContinuousNonlinearCorrelation1D    :attr:`~mfpml.problems.multi_fidelity_functions.ContinuousNonlinearCorrelation1D`
================================    ========================================================================================
