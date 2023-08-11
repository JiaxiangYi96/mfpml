Design of Experiments
=====================

we can create a design space by using :attr:`~mfpml.design_of_experiments.design_space.DesignSpace`.
The :attr:`~mfpml.design_of_experiments.design_space.DesignSpace` takes the following arguments: names, low_bound, high_bound, and types.

.. code-block:: python

   from mfpml.design_of_experiments.design_space import DesignSpace
   space = DesignSpace(names=['x1', 'x2'],
                    low_bound=[0.0, 0.0],
                    high_bound=[1.0, 1.0])
   >>> dict format
   design_space = space.design_space
   {'x1': [0.0, 1.0], 'x2': [0.0, 1.0]} 
   >>> numpy array format
   input_domain = space.input_domain
   [[0. 1.]
   [0. 1.]]

Single fidelity sampler
=======================

Example for single fidelity sampler
-----------------------------------

Single fidelity sampler by using :attr:`~mfpml.design_of_experiments.singlefideliy_samplers`.
Initialization of the sampler requires a :attr:`~mfpml.design_of_experiments.design_space.DesignSpace` and a seed.
Then we can get samples with :attr:`~mfpml.design_of_experiments.singlefideliy_samplers.SingleFidelitySampler.get_samples`.

.. code-block:: python

   from mfpml.design_of_experiments.singlefideliy_samplers import LatinHyperCube
   sampler = LatinHyperCube(design_space=design_space, seed=12)
   samples = sampler.get_samples(num_samples=10)
   >>>
   [[0.85791405 0.06944251]
   [0.38605036 0.98125293]
   [0.32699364 0.14201562]
   [0.91701757 0.80735379]
   [0.60346028 0.43486295]
   [0.13684348 0.52207483]
   [0.22297508 0.05505399]
   [0.50147379 0.89425058]
   [0.94087794 0.26977609]
   [0.28773614 0.68709683]]
.. note::
   you can use the above code block for other single fidelity samplers by 
   replacing :attr:`~mfpml.design_of_experiments.singlefideliy_samplers.LatinHyperCube` with
   :attr:`~mfpml.design_of_experiments.singlefideliy_samplers.RandomSampler` or
   :attr:`~mfpml.design_of_experiments.singlefideliy_samplers.SobolSequence` or
   :attr:`~mfpml.design_of_experiments.singlefideliy_samplers.FixNumberSampler`.

Implemented single fidelity samplers
------------------------------------

======================== ========================================================================================
Methods                   API of sampling methods                                            
======================== ========================================================================================         
LatinHyperCube             :attr:`~mfpml.design_of_experiments.singlefideliy_samplers.LatinHyperCube`                 
RandomSampler              :attr:`~mfpml.design_of_experiments.singlefideliy_samplers.RandomSampler`                     
SobolSequence              :attr:`~mfpml.design_of_experiments.singlefideliy_samplers.SobolSequence`    
FixNumberSampler           :attr:`~mfpml.design_of_experiments.singlefideliy_samplers.FixNumberSampler`                
======================== ========================================================================================


Multi fidelity sampler
======================

Example for multi fidelity sampler
----------------------------------

Multi fidelity sampler by using :attr:`~mfpml.design_of_experiments.multifidelity_samplers`.
Initialization of the sampler requires a :attr:`~mfpml.design_of_experiments.design_space.DesignSpace` and a seed.
Then we can get samples with :attr:`~mfpml.design_of_experiments.multifidelity_samplers.MultiFidelitySampler.get_samples`.

.. code-block:: python

   from mfpml.design_of_experiments.multifidelity_samplers import MFLatinHyperCube
   mf_sampler = MFSobolSequence(design_space=design_space,
                                seed=10,
                                nested=False)
   samples = mf_sampler.get_samples(num_lf_samples=4, num_hf_samples=2)
   >>> 
   {'hf': array([[0.84127845, 0.03307158],
       [0.04359769, 0.88905829]]), 'lf': array([[0.17129549, 0.73985035],
       [0.64001456, 0.39570647],
       [0.50126341, 0.54964036],
       [0.0327029 , 0.33054989]])}


.. note::
   The argument of *nested* is used to determine whether the low fidelity samples are nested in the high fidelity samples.
   However, the *nested* argument is only used for :attr:`~mfpml.design_of_experiments.multifidelity_samplers.MFSobolSequence` for now.
   The usage of *nested* argument for other multi fidelity samplers will be added in the future.


Implemented multi fidelity samplers
-----------------------------------

======================== ========================================================================================
Methods                   API of sampling methods
======================== ========================================================================================
MFLatinHyperCube           :attr:`~mfpml.design_of_experiments.multifidelity_samplers.MFLatinHyperCube`
MFSobolSequence            :attr:`~mfpml.design_of_experiments.multifidelity_samplers.MFSobolSequence`
======================== ========================================================================================
