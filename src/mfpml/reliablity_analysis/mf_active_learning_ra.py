from collections import OrderedDict
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from ..models.mf_scale_kriging import ScaledKriging
from .prob_evaluation import ProbEval


class MFActiveLearningRA(ProbEval):

    def __init__(self, problem: Any, factor: float = 0.0) -> None:

        self.problem = problem
        self.factor = factor

    def change_problem(self, problem: Any) -> None:

        self.problem = problem

    def _initialization(self) -> None:

        self.iter = 0
        # true probability of failure
        self.history_pf = []
        self.history_pf_cov = []
        # estimated probability of failure
        self.history_pf_hat = []
        self.history_pf_cov_hat = []
        # log samples of each iteration
        self.log = OrderedDict()
        # parameters
        self.search_x = None
        # other params
        self.params = {}
        # get input domain
        self.params['design_space'] = self.problem.input_domain
        self.params['cr'] = self.problem.cr

    def run_analysis(self,
                     surrogate: ScaledKriging,
                     init_x: dict,
                     init_y: dict,
                     learning_function: Any,
                     stopping_criterion: Any,
                     stopping_threshold: float,
                     num_mcs: int = 10**4,
                     seed: int = 123,
                     print_info: bool = True) -> None:
        # surrogate model
        self.surrogate = surrogate
        # learning function and stopping criterion
        self.learning_function = learning_function
        self.stopping_criterion = stopping_criterion
        # modidfy stopping threshold
        self._stopping_threshold(stopping_threshold=stopping_threshold)
        # search space
        self.num_mcs = num_mcs
        self.seed = seed
        # initialization
        self._initialization()
        self.gen = 1
        iter = 0
        # first run
        self._first_run(init_x=init_x,
                        init_y=init_y,
                        seed=self.seed,
                        num_mcs=self.num_mcs)
        # main loop
        while self.pf_cov_hat > 0.05 or iter < 1:
            # update search space
            self.num_mcs = self.num_mcs*self.gen
            self.search_x = self.problem.generate_multivariate_samples(
                num_mcs=self.num_mcs, seed=self.seed)
            # get stopping value
            self.stopping_value = self.stopping_criterion.stopping_value(
                surrogate=self.surrogate,
                search_x=self.search_x,
                iter=0)
            while self.stopping_value >= self.stopping_threshold:
                # update the next point
                update_x = learning_function.query(
                    mf_surrogate=self.surrogate,
                    search_x=self.search_x,
                    cost_ratio=self.problem.cr,
                    sample_x=self.surrogate.sample_xh)

                # get update y
                update_y = self.problem(update_x, factor=self.factor)
                # update surrogate
                self.surrogate.update_model(Xnew=update_x,
                                            Ynew=update_y)

                # update paras
                self._update_paras(update_x=update_x, update_y=update_y)
                # update iter
                iter = iter + 1
                # stopping value
                self.stopping_value = self.stopping_criterion.stopping_value(
                    surrogate=self.surrogate,
                    search_x=self.search_x,
                    iter=iter)
                # print info
                if print_info:
                    self.__print_info(iter=iter)
            # update cov
            pf, pf_cov, pf_hat, pf_hat_cov = self.prob_eval(
                num_mcs=self.num_mcs, seed=self.seed)
            # update the pf and pf_hat
            self.pf_hat = pf_hat
            self.pf_cov_hat = float(pf_hat_cov)
            self.gen = self.gen + 1
            self.__print_info(iter=iter)

    def _first_run(self,
                   init_x: dict,
                   init_y: dict,
                   num_mcs: int = 10**4,
                   seed: int = 123,
                   print_info: bool = True) -> None:

        # train the surrogate model
        self.init_x = init_x
        self.iniy_y = init_y
        self.surrogate.train(samples=init_x, responses=init_y)
        # cost
        self.params['n_hf'] = init_x['hf'].shape[0]
        self.params['n_lf'] = init_x['lf'].shape[0]
        self.params['cost'] = self.params['n_hf'] + \
            self.params['n_lf'] / self.params['cr']
        # get the initial estimate of probability of failure
        pf, pf_cov, pf_hat, pf_hat_cov = self.prob_eval(
            num_mcs=num_mcs, seed=seed)
        # update stopping value
        self.stopping_value = self.stopping_criterion.stopping_value(
            surrogate=self.surrogate,
            search_x=self.search_x,
            iter=0)
        # update the pf and pf_hat
        self.pf = pf
        self.pf_cov = pf_cov
        self.history_pf.append(pf)
        self.history_pf_cov.append(pf_cov)
        # update the pf_hat and pf_hat_cov
        self.pf_hat = pf_hat
        self.pf_cov_hat = float(pf_hat_cov)
        # update the history of pf and pf_hat
        self.history_pf_hat.append(pf_hat)
        self.history_pf_cov_hat.append(pf_hat_cov)
        self.log[0] = (init_x, init_y)
        if print_info:
            self.__print_info(iter=0)

    def _stopping_threshold(self, stopping_threshold: float) -> None:
        # modify stopping threshold
        if type(self.stopping_criterion).__name__ == 'UStoppingCriteria':
            self.stopping_threshold = -stopping_threshold
        else:
            self.stopping_threshold = stopping_threshold

    def _update_paras(self, update_x: dict,
                      update_y: dict) -> None:
        self.iter = self.iter + 1
        # update log
        self.log[self.iter] = (update_x, update_y)
        # update the number of samples
        if update_x['hf'] is not None:
            self.params['n_hf'] += update_x['hf'].shape[0]
        elif update_x['lf'] is not None:
            self.params['n_lf'] += update_x['lf'].shape[0]
        #
        self.params['cost'] = self.params['n_hf'] + \
            self.params['n_lf'] / self.params['cr']
        # update the pf and pf_hat
        pf, pf_cov, pf_hat, pf_hat_cov = self.prob_eval(
            num_mcs=self.num_mcs, seed=self.seed)
        # update the pf and pf_hat
        self.pf = pf
        self.pf_cov = pf_cov
        self.history_pf.append(pf)
        self.history_pf_cov.append(pf_cov)
        # update the pf_hat and pf_hat_cov
        self.pf_hat = pf_hat
        self.pf_cov_hat = float(pf_hat_cov)
        # update the history of pf and pf_hat
        self.history_pf_hat.append(pf_hat)
        self.history_pf_cov_hat.append(pf_hat_cov)

    def plot_fitting(self, save_fig: bool = False,
                     fig_name: str = "alkra.png", **kwargs):
        """plot the fitting of surrogate model
        """
        num_dim = self.problem._get_dimension
        num_plot = 200
        if num_dim == 2:
            # contour plot
            x1_plot = np.linspace(
                start=self.problem._input_domain[0, 0],
                stop=self.problem._input_domain[0, 1],
                num=num_plot,
            )
            x2_plot = np.linspace(
                start=self.problem._input_domain[1, 0],
                stop=self.problem._input_domain[1, 1],
                num=num_plot,
            )
            X1, X2 = np.meshgrid(x1_plot, x2_plot)
            Y = np.zeros([len(X1), len(X2)])
            pred_y = np.zeros([len(X1), len(X2)])
            # get the values of Y at each mesh grid
            for i in range(len(X1)):
                for j in range(len(X1)):
                    xy = np.array([X1[i, j], X2[i, j]])
                    xy = np.reshape(xy, (1, 2))
                    Y[i, j] = self.problem.hf(x=xy)
                    pred_y[i, j] = self.surrogate.predict(
                        xy, return_std=False)
            fig, ax = plt.subplots(**kwargs)
            # plot the real function
            # number init
            num_init_h = self.init_x['hf'].shape[0]
            num_init_l = self.init_x['lf'].shape[0]
            ax.plot(self.search_x[:, 0], self.search_x[:, 1],
                    '.', color="lightgray", label="MCS samples")
            # plot the surrogate model
            ax.plot(self.surrogate.sample_xh[:num_init_h, 0],
                    self.surrogate.sample_xh[:num_init_h, 1],
                    'bo', label='init hf samples')
            ax.plot(self.surrogate.sample_xl[:num_init_l, 0],
                    self.surrogate.sample_xl[:num_init_l, 1],
                    'b+', label='init lf samples')
            # plot added samples
            ax.plot(self.surrogate.sample_xh[num_init_h:-1, 0],
                    self.surrogate.sample_xh[num_init_h:-1, 1],
                    'r*', label='added hf samples')
            ax.plot(self.surrogate.sample_xl[num_init_l:-1, 0],
                    self.surrogate.sample_xl[num_init_l:-1, 1],
                    'r+', label='added lf samples')
            cs1 = ax.contour(
                X1, X2, Y, levels=[-10**6, 0, 10**6],
                colors='#EE6677', linewidths=2)
            cs2 = ax.contour(
                X1, X2, pred_y, levels=[-10**6, 0, 10**6],
                colors='#4477AA', linewidths=2)
            cs1.collections[1].set_label("real limit state")
            cs2.collections[1].set_label("predictive limit state")
            plt.xlabel(r"$x_1$", fontsize=12)
            plt.ylabel(r"$x_2$", fontsize=12)
            plt.legend(loc="upper right", fontsize=10)
            plt.grid()
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.5)
                ax.spines[axis].set_color('k')
            if save_fig is True:
                fig.savefig(fig_name, dpi=300, bbox_inches="tight")
            plt.show()

    def __print_info(self, iter: int) -> None:
        """print probability of failure

        Parameters
        ----------
        iter : int
            iteration number
        """
        # print the best y of current iteration
        print(
            f"=================== gen: {self.gen},"
            f"iter:{iter} ==================")
        print(f"estimated pf: {self.pf_hat:6f}",
              f"estimated cov: {self.pf_cov_hat:6f}")
        print(f"real pf: {self.pf:6f}",
              f"real cov: {self.pf_cov:6f}")
        print(f"hf samples: {self.params['n_hf']}",
              f"lf samples: {self.params['n_lf']}",
              f"cost: {self.params['cost']}",
              f"stopping value: {np.abs(self.stopping_value):6f}")
