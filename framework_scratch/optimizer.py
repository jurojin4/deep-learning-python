from typing import Any, List, Union, Tuple
from module import Activation_Module, Module, Sequential

import types

def _step_unimplemented(self, *epoch: Any) -> None:
    raise NotImplementedError(f'Optimizer[{type(self).__name__}] is missing the required "step function.')

class Optimizer:
    """
    Base optimizer class.
    """
    _name: str
    def __init__(self, model: Union[Module, Sequential], learning_rate: Union[List[Union[int, float]], Tuple[Union[int, float]], int, float], milestones: Union[List[int], Tuple[int], None] = None):
        """
        Initializes the Optimizer class.
        
        :param Model | Sequential **model**: Training model.
        :param List[int | float] | Tuple[int | float] | int | float **learning_rate**: Scalar(s) to apply during the gradient descent.
        :param List[int] | Tuple[int] | None, optional **milestones**: Milestones epochs where the learning rate is been changed.
        """
        assert (isinstance(model, Module) or isinstance(model, Sequential)) and not isinstance(model, Activation_Module), f"model has to be a {Module} or {Sequential} instance, not {type(model)}."
        self._model = model
        if hasattr(learning_rate, "__len__"):
            assert isinstance(milestones, (list, tuple)) and milestones is not None, f"milestones parameters has to be a {list} or {tuple} instance, not {type(milestones)}."
            assert len(learning_rate) == len(milestones) + 1, f"learning_rate has to be of length = len(milestones) + 1\nlen(learning_rate): {len(learning_rate)}\nlen(milestones): {len(milestones)}"
            self._has_learning_rates = True
            self._current_index = 0

            self._learning_rates = learning_rate
            self._milestones = milestones
        else:
            self._has_learning_rates = False
            self._learning_rate = learning_rate

        self.step = types.MethodType(_step_unimplemented, self)

    def _define_name(self, name: str, **kwargs) -> None:
        """
        Method that defines the optimizer name. Used by the __repr__ built-in Python method.

        :param str **name**: Optimizer Name.
        :param ****kwargs**: Arbitrary keyword arguments used to complete the optimizer name if necessary.
        """
        assert isinstance(name, str), f"name has to be a {str} instance, not {type(name)}."

        params_str = "("
        for enu, (name_value, value) in enumerate(kwargs.items()):
            params_str += name_value + "=" + str(value)
            if enu != len(kwargs) - 1:
                params_str += ", "
            else:
                params_str += ")"
        self._name = name + params_str

    def __repr__(self) -> str:
        """
        Built-in Python method that represents the string representation of Optimizer instance.

        :return: String representation of the optimizer.
        :rtype: str
        """
        return self._name

class SGD(Optimizer):
    """
    Stochastic Gradient Descent.
    """
    def __init__(self, model: Union[Module, Sequential],
                 learning_rate: Union[List[Union[int, float]], Tuple[Union[int, float]], int, float], 
                 milestones: Union[List[int], Tuple[int], None] = None):
        """
        Initializes the SGD class.
        
        :param Model | Sequential **model**: Training model.
        :param List[int | float] | Tuple[int | float] | int | float **learning_rate**: Scalar(s) to apply during the gradient descent.
        :param List[int] | Tuple[int] | None, optional **milestones**: Milestones epochs where the learning rate is been changed.
        """
        super().__init__(model, learning_rate, milestones)
        self._define_name(name="SGD", learning_rate=learning_rate, milestones=milestones)
        self.step = self._step

    def __step(self, module: Module, epoch: Union[int, None] = None) -> None:
        """
        Method that applies the stochastic gradient descent to the module.

        :param Module **module**: Module where to apply the descent.
        :param int | None, optional epoch: if _has_learning_rates equals true then epoch is used to change depending of epoch(s) milestones.
        """
        for key in module.parameters:
            if self._has_learning_rates:
                if self._current_index == len(self._milestones):
                    if epoch >= self._milestones[self._current_index]:
                        self._current_index += 1
                module.parameters[key].tensor = module.parameters[key].tensor - self._learning_rates[self._current_index] * module.parameters[key].grad
            else:
                module.parameters[key].tensor = module.parameters[key].tensor - self._learning_rate * module.parameters[key].grad

    def _step(self, epoch: Union[int, None] = None) -> None:
        """
        Method that applies the stochastic gradient descent to all modules.

        :param int | None, optional epoch: if _has_learning_rates equals true then epoch is used to change depending of epoch(s) milestones.
        """
        if isinstance(self._model, Sequential):
            for _, module in self._model:
                if isinstance(module, Activation_Module):
                    continue
                try:
                    self.__step(module, epoch)
                except Exception as exception:
                    print(exception)
                    continue
        else:
            try:
                self.__step(self._model, epoch)
            except Exception as exception:
                print(exception)