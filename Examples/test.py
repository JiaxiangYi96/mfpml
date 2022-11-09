
from collections import OrderedDict
from DoE.Samplers import LatinHyperCube

def main():
    # define the design space
    doe_variables = OrderedDict({'x': [0.0, 1.0]})
    # define number of samples
    num_points = 10
    # define the information of outputs
    name_outputs = ['y']

    doe_sampler = LatinHyperCube()
    samples = doe_sampler.Sampling(num_samples=num_points,
                                   design_space=doe_variables,
                                   out_names=name_outputs,
                                   seed=123456)
    print(samples)

    return samples

if __name__ == "__main__":
    main()