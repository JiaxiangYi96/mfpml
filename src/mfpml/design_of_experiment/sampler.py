# third-party
from abc import ABC


class Sampler(ABC):
    """
    Class for drawing samples from design space

    """

    def __init__(self, design_space: dict, seed: int = 123456):
        """
        Initialization of sampler class
        Parameters
        ----------
        design_space: dict
            design space
        seed: int

        """
        self.seed = seed
        self.design_space = design_space
        self.num_dim = len(design_space)
        self._samples = None

    def get_samples(self, num_samples: int, **kwargs) -> any:
        """
        Get the samples

        Parameters
        ----------
        num_samples: int
            number of samples
        kwargs: int,int
            num_lf_samples: int
            num_hf_samples: int

        Returns
        ---------
        samples: any
            samples

        Notes
        ---------
        The function should be completed at the sub-sclass


        """

        raise NotImplementedError("Subclasses should implement this method.")

    def _create_pandas_frame(self):
        """
        this function is used to create pandas framework for the doe
        the output will be added at the end of the pandas dataframe
        but without giving names

        Parameters
        ----------

        Returns
        -------
        """

        # load the number of outputs and corresponding names
        # transfer the variables to a pandas dataframe
        raise NotImplementedError("Subclasses should implement this method.")

    def save_doe(self, json_name: str = "doe") -> None:
        """
        This function is used to save the design_of_experiment to Json files

        Parameters
        ----------
        json_name:str
            name for the Json.file

        Returns
        -------

        """

        self._samples.to_json(json_name + ".json", index=True)

    def plot_samples(
        self, fig_name: str = "figure", save_fig: bool = False
    ) -> None:
        """
        function to visualize the one dimension design of experiment
        Parameters
        ----------
        fig_name: str
            name of the figure file
        save_fig: bool
            operator to claim save the figure or not

        Returns
        -------

        """
        raise NotImplementedError("Subclasses should implement this method.")
