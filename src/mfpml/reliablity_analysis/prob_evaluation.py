from typing import Any

import numpy as np


class ProbEval:
    """a class for evaluating the probability of failure during ALKRA process
    """

    def prob_eval(self,
                  num_mcs: int = 10**6,
                  seed: int = 123) -> tuple[Any, Any, float, float]:
        """
        evaluate the probability of failure

        Parameters
        ----------
        num_mcs : int, optional
            number of Monte Carlo samples, by default 10**6
        seed : int, optional
            random seed, by default 123
        """

        # get the search_x
        self.search_x = \
            self.problem.generate_multivariate_samples(num_mcs=num_mcs,
                                                       seed=seed)
        # evaluate the limit state function
        pf, pf_var = self.problem.prob_cal(search_x=self.search_x)

        # evaluate the surrogate model
        pf_hat, pf_hat_var = self._prob_eval_sur()

        return pf, pf_var, pf_hat, pf_hat_var

    def _prob_eval_sur(self) -> tuple[float, float]:
        """evaluate the probability of failure using surrogate model
        """
        # get pf
        pred_y = self.surrogate.predict(self.search_x, return_std=False)
        # calculate the probability of failure
        prob_fail = float(np.sum(pred_y <= 0) / len(pred_y))
        # cov
        cov_fail = np.sqrt((1 - prob_fail) / (len(pred_y)*prob_fail))
        return prob_fail, cov_fail
