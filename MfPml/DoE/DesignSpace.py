from MfPml.Core.Space import DesignSpace


class CreateDesignSpace(DesignSpace):

    def __init__(self) -> None:
        """Assemble the design space sequentially
        """
        self.names = list()
        self.lower_bound = list()
        self.higher_bound = list()
        self.design_space = {}

    def add_variables(self, name: list, lower_bound: list, high_bound: list) -> None:
        """
        Add variables and corresponding design space sequentially
        Note:  one or more dimensional could be added to the design space sequentially

        Parameters
        ----------
        name : list
            Names of added variables
        lower_bound : list
            Lower bound of added design variables
        high_bound :  list
            Higher bound of added design variables
        Returns
        -------
        None

        """

        if len(self.names) == 0:
            self.names = name
            self.lower_bound = lower_bound
            self.higher_bound = high_bound
        else:
            self.names.extend(name)
            self.lower_bound.extend(lower_bound)
            self.higher_bound.extend(high_bound)

    def assemble_design_space(self) -> dict:
        """Function to assemble the design space based on the design
        variable information defined in AddVaribales function


        Returns
        -------
        design_space: dict
            design space generated for the objective problem

        """

        for ii, name in enumerate(self.names):
            self.design_space[name] = [self.lower_bound[ii], self.higher_bound[ii]]

        return self.design_space
