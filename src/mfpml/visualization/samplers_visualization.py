from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np

# import local functions
from mfpml.design_of_experiment.singlefideliy_samplers import SobolSequence, LatinHyperCube, RandomSampler
from mfpml.design_of_experiment.multifidelity_samplers import MFLatinHyperCube, MFSobolSequence
class SFSamplerVisualization:
    """Sampler visualization class
    """

    def __init__(self, sampler: SobolSequence|LatinHyperCube|RandomSampler):

        self.sampler = sampler

        # check if the samples are generated (self.samples is not None)
        if self.sampler.samples is None:
            raise ValueError("Please generate the samples first.")


    def plot_samples(self, save_figure: bool = False, filename: str = "samples.png"):
         
        # check the fimension of the samples
        num_dim = self.sampler.num_dim
        if num_dim == 1:
            self._plot_1D(self.sampler.samples, save_figure, filename)
        elif num_dim == 2:
            self._plot_2D(self.sampler.samples, save_figure, filename)
        elif num_dim == 3:
            self._plot_3D(self.sampler.samples, save_figure, filename)
        else:
            raise ValueError("The number of dimensions is not supported")
    
    def _plot_2D(self, samples: np.ndarray, save_figure: bool, filename: str):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(samples[:, 0], samples[:, 1], c='#EE7733', marker='*',s=50, label='samples')
        plt.xlabel(r"$x_1$", fontsize=12)
        plt.ylabel(r"$x_2$", fontsize=12)
        plt.title("2D samples", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid()
        # set the font size of the ticks
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # set the line width of axes    
        plt.gca().spines['top'].set_linewidth(1.5)
        plt.gca().spines['right'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.gca().spines['left'].set_linewidth(1.5)
        if save_figure:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def _plot_3D(self, samples: np.ndarray, save_figure: bool, filename: str):
        
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(samples[:, 0],
                   samples[:, 1],
                   samples[:, 2],
                   c='#EE7733',
                   marker='*',s=50, label='samples')
        ax.set_xlabel(r"$x_1$", fontsize=12)
        ax.set_ylabel(r"$x_2$", fontsize=12)
        ax.set_zlabel(r"$x_3$", fontsize=12)
        plt.title("3D samples", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid()
        # set the font size of the ticks
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # set the line width of axes
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)

        if save_figure:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_1D(self, samples: np.ndarray, save_figure: bool, filename: str):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(samples[:, 0], np.zeros(samples.shape[0]), c='#EE7733', marker='*',s=50, label='samples')
        plt.xlabel(r"$x_1$", fontsize=12)
        plt.title("1D samples", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid()
        # set the font size of the ticks
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # set the line width of axes
        plt.gca().spines['top'].set_linewidth(1.5)
        plt.gca().spines['right'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.gca().spines['left'].set_linewidth(1.5)
        if save_figure:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class MFSamplerVisualization:

    def __init__(self, sampler: MFLatinHyperCube|MFSobolSequence):

        self.sampler = sampler
        # check if the data is an attribute of the sampler
        assert hasattr(self.sampler, 'data'), "Please generate the samples first."

        # get the dimension of the samples
        self.num_dim = self.sampler.num_dim
        # get the fidelity of the samples
        self.num_fidelity = self.sampler.num_fidelity

    
    def plot_samples(self, save_figure: bool = False, filename: str = "samples.png"):
        num_dim = self.sampler.num_dim
        if num_dim == 1:
            self._plot_1D(self.sampler.data, save_figure, filename)
        elif num_dim == 2:
            self._plot_2D(self.sampler.data, save_figure, filename)
        elif num_dim == 3:
            self._plot_3D(self.sampler.data, save_figure, filename)
        else:
            raise ValueError("The number of dimensions is not supported")
    
    def _plot_2D(self, data: List, save_figure: bool, filename: str):
        
        fig, ax = plt.subplots(figsize=(5, 4))
        # plot the samples
        for fidelity, samples in reversed(list(enumerate(data))):
            ax.scatter(samples[:, 0],
                       samples[:, 1],
                       c=self.__get_color(fidelity),
                       marker=self.__get_marker(fidelity), 
                       s=(30+ 10*fidelity), 
                       label=f'fidelity {fidelity}')
            
        plt.xlabel(r"$x_1$", fontsize=12)
        plt.ylabel(r"$x_2$", fontsize=12)
        plt.title("2D samples", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid()
        # set the font size of the ticks
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # set the line width of axes
        plt.gca().spines['top'].set_linewidth(1.5)
        plt.gca().spines['right'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.gca().spines['left'].set_linewidth(1.5)
        if save_figure:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def _plot_3D(self, samples: np.ndarray, save_figure: bool, filename: str):
        
        fig, ax = plt.subplots(figsize=(5, 4))
        # plot the samples
        ax = fig.add_subplot(111, projection='3d')
        for fidelity, samples in reversed(list(enumerate(samples))):
            ax.scatter(samples[:, 0],
                       samples[:, 1],
                       samples[:, 2], 
                       c=self.__get_color(fidelity),
                       marker=self.__get_marker(fidelity),
                        s=(30+ 10*fidelity), 
                        label=f'fidelity {fidelity}')
        ax.set_xlabel(r"$x_1$", fontsize=12)
        ax.set_ylabel(r"$x_2$", fontsize=12)
        ax.set_zlabel(r"$x_3$", fontsize=12)
        plt.title("3D samples", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid()
        # set the font size of the ticks
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # set the line width of axes
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)

        if save_figure:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_1D(self, samples: np.ndarray, save_figure: bool, filename: str):
        
        fig, ax = plt.subplots(figsize=(5, 4))
        # plot the samples
        for fidelity, samples in reversed(list(enumerate(samples))):
            ax.scatter(samples[:, 0], np.zeros(samples.shape[0]), c=self.__get_color(fidelity), marker=self.__get_marker(fidelity), s=(30+ 10*fidelity), label=f'fidelity {fidelity}')
        plt.xlabel(r"$x_1$", fontsize=12)
        plt.title("1D samples", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid()
        # set the font size of the ticks
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # set the line width of axes
        plt.gca().spines['top'].set_linewidth(1.5)
        plt.gca().spines['right'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.gca().spines['left'].set_linewidth(1.5)
        if save_figure:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def __get_color(self, fidelity: int) -> str:
        colors = ['#EE7733', '#0077BB', '#009988', '#EE3377', '#CC3311']
        return colors[fidelity]
    
    def __get_marker(self, fidelity: int) -> str:
        markers = ['+','o','v', '^', 's', 'D']
        return markers[fidelity]
    

# test the mf sampler visualization
if __name__ == '__main__':
    # create a MFLatinHyperCube sampler
    mf_sampler = MFLatinHyperCube(design_space=np.array([[0, 1], [0, 1]]), num_fidelity=2, nested=True)
    # generate samples
    mf_sampler.get_samples(num_samples=[10, 20], seed=0)
    # create a MFSamplerVisualization object
    mf_visualizer = MFSamplerVisualization(mf_sampler)
    # plot the samples
    mf_visualizer.plot_samples(save_figure=False, filename="mf_samples.png")