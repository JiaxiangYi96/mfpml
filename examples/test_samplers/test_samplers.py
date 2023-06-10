import os
import sys

# import local libraries
# path of local project
folder_path = "/home/yaga/Documents/GitHub/mfpml"
sys.path.insert(0, folder_path)
# import the mfpml package
from mfpml.design_of_experiment.sf_samplers import SobolSequence
from mfpml.design_of_experiment.space import DesignSpace

# define the design space
space = DesignSpace(
    names=["x1", "x2"], low_bound=[0.0, 0.0], high_bound=[1.0, 1.0]
)
print(space.design_space)
# output the deisgn space in a dict format
design_space = space.design_space
print(f"dict format: \n {design_space}")
# output the design in a array format
input_domain = space.input_domain
print(f"array format: \n {input_domain}")
# test the sf SobolSequence sampler
sampler = SobolSequence(design_space=design_space, seed=12)
samples = sampler.get_samples(num_samples=10)

sampler.plot_samples(fig_size=(5, 4))
