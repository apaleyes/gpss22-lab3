"""
This file provides a couple of plotting routines that resemble GPyOpt APIs.
"""

import numpy as np
import matplotlib.pyplot as plt
from emukit.core import ParameterSpace
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop



def plot_acquisition(emukit_bo_loop: BayesianOptimizationLoop, space: ParameterSpace,
                     label_x: str=None, label_y: str=None):
    """
    Plots the model and the acquisition function.
        if self.input_dim = 1: Plots data, mean and variance in one plot and the acquisition function in another plot
        if self.input_dim = 2: as before but it separates the mean and variance of the model in two different plots
    :param label_x: Graph's x-axis label, if None it is renamed to 'x' (1D) or 'X1' (2D)
    :param label_y: Graph's y-axis label, if None it is renamed to 'f(x)' (1D) or 'X2' (2D)
    """
    return _plot_acquisition(space.get_bounds(),
                            emukit_bo_loop.loop_state.X.shape[1],
                            emukit_bo_loop.model,
                            emukit_bo_loop.model.X,
                            emukit_bo_loop.model.Y,
                            emukit_bo_loop.candidate_point_calculator.acquisition,
                            emukit_bo_loop.get_next_points(None),
                            label_x,
                            label_y)

def _plot_acquisition(bounds, input_dim, model, Xdata, Ydata, acquisition_function, suggested_sample,
                     label_x=None, label_y=None, color_by_step=True):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''

    if input_dim ==1:
        # Plots for 1D input
        if not label_x:
            label_x = 'x'

        if not label_y:
            label_y = 'f(x)'

        x_grid = np.arange(bounds[0][0], bounds[0][1], 0.001)
        x_grid = x_grid.reshape(len(x_grid),1)
        acq = acquisition_function.evaluate(x_grid)
        acq_normalized = (acq - min(acq))/(max(acq) - min(acq))
        m, v = model.predict(x_grid)

        upper_conf_bound = m[:, 0] + 1.96 * np.sqrt(v)[:, 0]
        lower_conf_bound = m[:, 0] - 1.96 * np.sqrt(v)[:, 0]

        plt.fill_between(x_grid[:, 0], upper_conf_bound, lower_conf_bound, color='b', alpha=0.2)
        plt.plot(x_grid, lower_conf_bound, 'k-', alpha=0.2)
        plt.plot(x_grid, upper_conf_bound, 'k-', alpha=0.2)
        plt.plot(x_grid, m, 'k-', lw=1, alpha=0.6)

        plt.plot(Xdata, Ydata, 'r.', markersize=10)
        plt.axvline(x=suggested_sample[len(suggested_sample)-1], color='r')
        factor = max(upper_conf_bound) - min(lower_conf_bound)

        plt.plot(x_grid, 0.2*factor*acq_normalized - abs(min(lower_conf_bound)) - 0.25*factor, 'r-',
                 lw=2, label='Acquisition')
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.ylim(min(lower_conf_bound) - 0.25*factor,  max(upper_conf_bound) + 0.05*factor)
        plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        plt.legend(loc='upper left')

    elif input_dim == 2:

        if not label_x:
            label_x = 'X1'

        if not label_y:
            label_y = 'X2'

        n = Xdata.shape[0]
        colors = np.linspace(0, 1, n)
        cmap = plt.cm.Reds
        norm = plt.Normalize(vmin=0, vmax=1)
        points_var_color = lambda X: plt.scatter(
            X[:,0], X[:,1], c=colors, label=u'Observations', cmap=cmap, norm=norm)
        points_one_color = lambda X: plt.plot(
            X[:,0], X[:,1], 'r.', markersize=10, label=u'Observations')
        X1 = np.linspace(bounds[0][0], bounds[0][1], 200)
        X2 = np.linspace(bounds[1][0], bounds[1][1], 200)
        x1, x2 = np.meshgrid(X1, X2)
        X = np.hstack((x1.reshape(200*200,1),x2.reshape(200*200,1)))
        acq = acquisition_function.evaluate(X)
        acq_normalized = (acq - min(acq))/(max(acq) - min(acq))
        acq_normalized = acq_normalized.reshape((200,200))
        m, v = model.predict(X)
        plt.figure(figsize=(15,5))
        plt.subplot(1, 3, 1)
        plt.contourf(X1, X2, m.reshape(200,200),100)
        plt.colorbar()
        if color_by_step:
            points_var_color(Xdata)
        else:
            points_one_color(Xdata)
        plt.ylabel(label_y)
        plt.title('Posterior mean')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        ##
        plt.subplot(1, 3, 2)
        plt.contourf(X1, X2, np.sqrt(v.reshape(200,200)),100)
        plt.colorbar()
        if color_by_step:
            points_var_color(Xdata)
        else:
            points_one_color(Xdata)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title('Posterior sd.')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        ##
        plt.subplot(1, 3, 3)
        plt.contourf(X1, X2, acq_normalized,100)
        plt.colorbar()
        plt.plot(suggested_sample[:,0],suggested_sample[:,1],'m.', markersize=10)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title('Acquisition function')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))

    else:
        raise ValueError(f'Cannot plot inputs higher than 2D, {input_dim} given')

def plot_convergence(X, Y):
    '''
    Plots the convergence of standard Bayesian optimization algorithms.

    :param X: Locations of evaluated points
    :param Y: Results of evaluations

    '''
    n = X.shape[0]

    ## Distances between consecutive x's
    aux = (X[1:n,:] - X[0:n-1,:])**2
    distances = np.sqrt(aux.sum(axis=1))

    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(n-1)), distances, '-ro')
    plt.xlabel('Iteration')
    plt.ylabel('d(x[n], x[n-1])')
    plt.title('Distance between consecutive x\'s')

    # Best found value at each iteration
    best_Y = np.minimum.accumulate(Y)

    plt.subplot(1, 2, 2)
    plt.plot(list(range(n)), best_Y, '-o')
    plt.title('Value of the best selected sample')
    plt.xlabel('Iteration')
    plt.ylabel('Best y')

