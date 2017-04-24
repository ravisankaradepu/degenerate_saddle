import theano
import theano.tensor as T
import numpy as np
import theano.sandbox.rng_mrg as RNG_MRG
from theano import function

from neupy.utils import asfloat
from neupy.core.properties import ProperFractionProperty, Property
from .base import MinibatchGradientDescent
import numpy
rng = numpy.random.RandomState(1234)
mrg = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(2**30))

__all__ = ('noisy',)


class noisy(MinibatchGradientDescent):
    """
    Momentum algorithm.

    Parameters
    ----------
    momentum : float
        Control previous gradient ratio. Defaults to ``0.9``.

    nesterov : bool
        Instead of classic momentum computes Nesterov momentum.
        Defaults to ``False``.

    {MinibatchGradientDescent.Parameters}

    Attributes
    ----------
    {MinibatchGradientDescent.Attributes}

    Methods
    -------
    {MinibatchGradientDescent.Methods}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> mnet = algorithms.Momentum((2, 3, 1))
    >>> mnet.train(x_train, y_train)

    See Also
    --------
    :network:`GradientDescent` : GradientDescent algorithm.
    """
    momentum = ProperFractionProperty(default=0.9)
    nesterov = Property(default=False, expected_type=bool)

    def init_param_updates(self, layer, parameter):
        step = self.variables.step

        parameter_shape = parameter.get_value().shape

        gradient = T.grad(self.variables.error_func, wrt=parameter)
	previous_velocity = theano.shared(
            name="{}/previous-velocity".format(parameter.name),
            value=asfloat(np.zeros(parameter_shape)),
        )

	noise = mrg.uniform(low=-0.001, high=0.001, size=gradient.shape, dtype=theano.config.floatX) 
	velocity = self.momentum * previous_velocity - step * (gradient+noise)

        if self.nesterov:
            velocity = self.momentum * velocity - step * gradient

       
        return [
            (parameter, parameter+velocity),
            (previous_velocity, velocity),
        ]

