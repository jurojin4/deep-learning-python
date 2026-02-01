from module import Activation_Module, Module, Parameters
from typing import Literal, Union

import math
import tensyx as ts

class Linear(Module):
    """
    Module that implements a neural layer. For a Tensor Input x:\n
    :math:`Z = x @ W + b`\n

    **Shapes**
    - Z: (batch_size, out_features)
    - x: (batch_size, in_features)\n
    - W: (in_features, out_features)\n
    - b: (1, out_features)\n

    **remark**: When the output of `x @ W` is added to b, we got different shapes (batch_size, out_features) and (1, out_features),
    but numpy uses the broadcasting to clone the line of b on the first dimension to have equal shapes.

    Attributes:
        biais: boolean, if equals True then `Z = x @ W + b` otherwise `Z = x @ W`.
        parameters: Instance of Parameters class. Contains the different parameters:
                    - parameters["W"]
                    - parameters["b"]\n

                    Those parameters are initialized depending on the chosen method:
                    - normal distribution: weights and biais take value in `[]`.
                    - uniform distribution: weights and biais take value in `[-in_features;in_features]`.
                    - xavier distribution: weights and biais take value in `[-\sqrt{k};\sqrt{k}]` where `k = \frac{1}{in_features}`.
                    - if none is chosen then randomly initialized.

    Examples:
        >>> layer = Linear(6, 8)
        >>> x = tensyx.random.rand((2, 6))
        >>> y = layer(x)
    """
    def __init__(self, in_features: int, out_features: int, biais: bool = True, initialize_method: Union[Literal["normal", "uniform", "xavier"], None] = "xavier") -> None:
        """
        Initializes the Linear class.

        :param int, optional **in_features**: Strictly positive integer representing the vector input size.
        :param int, optional **out_features**: Strictly positive integer representing the vector output size and also refers (firstly) to the number of neurons.
        :param boolean, optional **biais**: If this is set true, all neurons will have a biais with their weighted sum.
        :param str, optional **initialize_method**: Initializes the weights (and biais) following a specified probability distribution ("normal", "uniform" or "xavier"). If equals None, it's randomly initialized. Set to `xavier`.
        """
        super().__init__()
        assert isinstance(in_features, int) and in_features > 0, f"in_features has to be a strictly positive integer, not {type(in_features)} | in_features: {in_features}."
        assert isinstance(out_features, int) and out_features > 0, f"out_features has to be a strictly positive integer, not {type(out_features)} | out_features: {out_features}."
        assert isinstance(biais, bool), f"biais has to be a {bool} instance, not {type(biais)}."
        assert isinstance(initialize_method, str) or initialize_method is None, f"initialize_method has to be a {str} instance or equals to {None}, not {type(initialize_method)}."

        # Weights Initialization
        self.biais = biais
        self._initialize_method = initialize_method
        self._initialize_wand(in_features, out_features, initialize_method)

        # Methods Definition
        self.forward = self._forward
        self.forward_with_specified_parameters = self._forward_with_specified_parameters
        self.count_parameters = self._count_parameters
        self.reinitialize = self._reinitialize

        # Name Definition
        self._define_name(name=self.__class__.__name__, in_features=in_features, out_features=out_features, biais=biais)

    def _initialize_wand(self, in_features: int, out_features: int, initialize_method: Union[Literal["normal", "uniform", "xavier"], None] = "xavier") -> None:
        """
        Method that initializes the Weights and biais of the linear layer.

        :param int **in_features**: Strictly positive integer representing the vector input size.
        :param int **out_features**: Strictly positive integer representing the vector output size and also refers (firstly) to the number of neurons.
        :param str, optional **initialize_method**: Initializes the weights (and biais) following a specified probability distribution ("normal", "uniform" or "xavier"). If equals None, it's randomly initialize. Set to `xavier`.
        """
        if initialize_method == "normal":
            self.parameters.add(W = ts.random.normal(0, 1, size=(in_features, out_features)))
            if self.biais:
                self.parameters.add(b = ts.random.normal(0, 1, size=(1, out_features)))
        elif initialize_method == "uniform":
            self.parameters.add(W = ts.random.uniform(-in_features, in_features, size=(in_features, out_features)))
            if self.biais:
                self.parameters.add(b = ts.random.uniform(-in_features, in_features, size=(1, out_features)))
        else:
            if initialize_method == "xavier":
                k = math.sqrt(1 / in_features)
                self.parameters.add(W = ts.random.uniform(-k, k, (in_features, out_features)))
                if self.biais:
                    self.parameters.add(b = ts.random.uniform(-k, k, (1, out_features)))
            else:
                self.parameters.add(W = ts.random.rand((in_features, out_features)))
                if self.biais:
                    self.parameters.add(b = ts.random.rand(size=(1, out_features)))

    def _forward(self, x: ts.Tensor) -> ts.Tensor:
        """
        Method that forwards through the linear layer an input Tensor (Vector).

        :param Tensor **x**: Input tensor (vector) of size (batch_size, in_features).
        :return: Vector of size (batch_size, out_features).
        :rtype: Tensor
        """
        assert isinstance(x, (ts.Tensor, ts.np.ndarray)), f"x has to be a {ts.Tensor} instance or {ts.np.ndarray}, not {type(x)}."

        if self.biais:
            return x @ self.parameters["W"] + self.parameters["b"]
        else:
            return x @ self.parameters["W"]
        
    def _forward_with_specified_parameters(self, x: ts.Tensor, parameters: dict[str, ts.Tensor]) -> ts.Tensor:
        """
        Method that forwards through the linear layer an input Tensor (Vector) with specified parameters.

        :param Tensor **x**: Input tensor (vector) of size (batch_size, in_features).
        :param dict[str, ts.Tensor] **parameters**: Specified parameters which will be used to forward through the linear layer with the input tensor.
        :return: Vector of size (batch_size, out_features).
        :rtype: Tensor
        """
        assert isinstance(x, ts.Tensor), f"x has to be a {ts.Tensor} instance, not {type(x)}."
        assert isinstance(parameters, dict), f"parameters has to be a {dict} instance, not {type(parameters)}."

        if self.biais:
            return x @ parameters["W"] + parameters["b"]
        else:
            return x @ parameters["W"]
        
    def _count_parameters(self) -> int:
        """
        Method that returns the number of learnable parameters.

        :return: Number of learnable parameters.
        :rtype: int
        """
        num_parameters = 0
        for parameters_name in self.parameters:
            shape = self.parameters[parameters_name].shape
            num_parameters += shape[0] * shape[1]
        return num_parameters
    
    def _reinitialize(self) -> None:
        """
        Method that reinitializes weights and biais.
        """
        in_features, out_features = self.parameters["W"].shape
        del self.parameters
        super().__setattr__("parameters", Parameters())
        self._initialize_wand(in_features, out_features, self._initialize_method)    

class Identity(Activation_Module):
    """Identity Activation Function"""
    def __init__(self):
        """
        Initializes the Identity class.
        """
        super().__init__()
        self.forward = self._forward
        self._define_name(self.__class__.__name__)

    def _forward(self, x: ts.Tensor) -> ts.Tensor:
        """
        Method that forwards through the activation function an input Tensor.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        return ts.identity(x)

class Heaviside(Activation_Module):
    """Heaviside Activation Function"""
    def __init__(self):
        """
        Initializes the Heaviside class.
        """
        super().__init__()
        self.forward = self._forward
        self._define_name(self.__class__.__name__)

    def _forward(self, x: ts.Tensor) -> ts.Tensor:
        """
        Method that forwards through the activation function an input Tensor.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        return ts.heaviside(x)
    
class TanH(Activation_Module):
    """TanH Activation Function"""
    def __init__(self):
        """
        Initializes the TanH class.
        """
        super().__init__()
        self.forward = self._forward
        self._define_name(self.__class__.__name__)

    def _forward(self, x: ts.Tensor) -> ts.Tensor:
        """
        Method that forwards through the activation function an input Tensor.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        return ts.tanh(x)

class Sigmoid(Activation_Module):
    """Sigmoid Activation Function"""
    def __init__(self):
        """
        Initializes the Sigmoid class.
        """
        super().__init__()
        self.forward = self._forward
        self._define_name(self.__class__.__name__)

    def _forward(self, x: ts.Tensor) -> ts.Tensor:
        """
        Method that forwards through the activation function an input Tensor.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        return ts.sigmoid(x)
    
class ReLU(Activation_Module):
    """ReLU Activation Function"""
    def __init__(self):
        """
        Initializes the ReLU class.
        """
        super().__init__()
        self.forward = self._forward
        self._define_name(self.__class__.__name__)

    def _forward(self, x: ts.Tensor) -> ts.Tensor:
        """
        Method that forwards through the activation function an input Tensor.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        return ts.relu(x)
    
class Softmax(Activation_Module):
    """Softmax Activation Function"""
    def __init__(self, dim = 1):
        """
        Initializes the Softmax class.

        :param int **dim**: Dimension position where to apply the softmax.
        """
        super().__init__()
        self._dim = dim
        self.forward = self._forward
        self._define_name(self.__class__.__name__)

    def _forward(self, x: ts.Tensor) -> ts.Tensor:
        """
        Method that forwards through the activation function an input Tensor.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        return ts.softmax(x, self._dim)
    
class Sin(Activation_Module):
    """Sin Activation Function"""
    def __init__(self):
        """
        Initializes the Sin class.
        """
        super().__init__()
        self.forward = self._forward
        self._define_name(self.__class__.__name__)

    def _forward(self, x: ts.Tensor) -> ts.Tensor:
        """
        Method that forwards through the activation function an input Tensor.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        return ts.sin(x)