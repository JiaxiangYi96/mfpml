class DesignSpace:

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

        pass

    def assemble_design_space(self) -> dict:
        """Function to assemble the design space based on the design
        variable information defined in AddVaribales function


        Returns
        -------
        design_space: dict
            design space generated for the objective problem

        """

        pass
