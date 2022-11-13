# import numpy as np
# from matplotlib import pyplot as plt
#
#
# def plot_1d(data: np.ndarray) -> None:
#     """
#     Visualize the 1D case, set the y axis as Zero
#     Parameters
#     ----------
#     data : original data for visualization
#
#     Returns
#     -------
#
#     """
#     data = data.reshape((-1, 1))
#     with plt.style.context(['ieee', 'science']):
#         fig, ax = plt.subplots()
#         ax.plot(data, np.zeros((data.shape[0], 1)), '*', label='Samples')
#         ax.legend()
#         ax.set(xlabel=r'$x_1$')
#         ax.set(ylabel=r'$y$')
#         ax.autoscale(tight=True)
#         plt.show()
#         if save is True:
#             fig.savefig(name, dpi=300)
#
#
# def plot_2d(data: np.ndarray) -> None:
#     data = data.reshape((-1, 2))
#     with plt.style.context(['ieee', 'science']):
#         fig, ax = plt.subplots()
#         ax.plot(data, np.zeros((data.shape[0], 1)), '*', label='Samples')
#         ax.legend()
#         ax.set(xlabel=r'$x_1$')
#         ax.set(ylabel=r'$y$')
#         ax.autoscale(tight=True)
#         plt.show()
#         if save is True:
#             fig.savefig(name, dpi=300)
