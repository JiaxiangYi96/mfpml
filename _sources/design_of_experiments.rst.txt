
This section introduces **single-fidelity** and **multi-fidelity samplers** provided in the `mfpml` package. These methods efficiently generate sample points for design spaces in experimental studies, surrogate modeling, and reliability analysis.

Single-Fidelity Samplers
------------------------

The single-fidelity samplers generate sample points at a single resolution. Below is a table summarizing the available samplers:

========================  ========================================================================================
**Sampler**               **API of Sampling Methods**
========================  ========================================================================================
Sobol Sequence            :attr:`~mfpml.design_of_experiment.sf_samplers.SobolSequence`
Latin Hypercube           :attr:`~mfpml.design_of_experiment.sf_samplers.LatinHyperCube`
Random Sampler            :attr:`~mfpml.design_of_experiment.sf_samplers.RandomSampler`
Fix Number Sampler        :attr:`~mfpml.design_of_experiment.sf_samplers.FixNumberSampler`
========================  ========================================================================================

**Example Usage**:

.. code:: python

   import numpy as np
   from mfpml.design_of_experiment.sf_samplers import SobolSequence

   design_space = np.array([[0, 1], [0, 1]])
   sampler = SobolSequence(design_space=design_space)
   samples = sampler.get_samples(num_samples=5, seed=1)
   print(samples)

**Output**:

::

   [[0.5    0.5   ]
    [0.25   0.75  ]
    [0.75   0.25  ]
    [0.375  0.625 ]
    [0.875  0.125 ]]



Multi-Fidelity Samplers
-----------------------

Multi-fidelity samplers generate samples at multiple fidelity levels. Below is a summary of the available samplers:

========================  ========================================================================================
**Sampler**               **API of Sampling Methods**
========================  ========================================================================================
Multi-Fidelity Sobol      :attr:`~mfpml.design_of_experiment.mf_samplers.MFSobolSequence`
Multi-Fidelity LHS        :attr:`~mfpml.design_of_experiment.mf_samplers.MFLatinHyperCube`
========================  ========================================================================================


**Example Usage**:

.. code:: python

   from mfpml.design_of_experiment.mf_samplers import MFSobolSequence

   design_space = np.array([[0, 1], [0, 1]])
   mf_sampler = MFSobolSequence(design_space=design_space, num_fidelity=2, nested=False)
   samples = mf_sampler.get_samples(num_samples=[3, 5])
   print(samples)

**Output**:

::

   Fidelity 1:
   [[0.5    0.5   ]
    [0.25   0.75  ]
    [0.75   0.25  ]]

   Fidelity 2:
   [[0.5    0.5   ]
    [0.25   0.75  ]
    [0.75   0.25  ]
    [0.375  0.625 ]
    [0.875  0.125 ]]



Summary
-------

The samplers in `mfpml` enable efficient design of experiments for both single-fidelity and multi-fidelity cases. The attribute tables above summarize the available methods, helping users quickly select and apply the appropriate sampler for their needs.
