from typing import Any, Dict, Iterator, List, Tuple, Union

import types
import copy
import tensyx as ts

class Parameters:
    """
    Base module parameters class.
    """
    __parameters: dict[str, ts.Tensor]
    def __init__(self, **kwargs):
        """
        Initializes the Parameters class.

        :param ****kwargs**: Arbitrary keyword arguments for model parameters names.
        """
        super().__init__()
        super().__setattr__("__parameters", kwargs)

    def add(self, **kwargs) -> None:
        """
        Adds to the private variable dict[str, ts.Tensor] "__parameters".

        :param ****kwargs**: Arbitrary keyword arguments containing model parameters names with their respective tensor (ts.Tensor) (item).
        """
        for key, item in kwargs.items():
            assert isinstance(key, str), f"key has to be a {str} instance, not {type(key)}."
            assert isinstance(item, ts.Tensor), f"item has to be a {ts.Tensor} instance, not {type(item)}."
            self.__getattribute__("__parameters")[key] = item

    def keys(self):
        """
        Returns the keys of the private variable dict[str, ts.Tensor] "__parameters", parameters names.
        """
        return self.__getattribute__("__parameters").keys()

    def __len__(self) -> int:
        """
        Built-in Python method that returns the length of the private dictionary "__parameters".

        :return: Number of samples in the dataset.
        :rtype: int
        """
        return len(self.__getattribute__("__parameters"))
    
    def __iter__(self) -> Iterator:
        """
        Built-in Python method that returns an iterator of the private dictionary "__parameters".

        :return: A key iterator.
        :rtype: Iterator
        """
        return iter(self.__getattribute__("__parameters"))

    def __getitem__(self, key: str) -> ts.Tensor:
        """
        Get the specified item according the key in the private dictionary "__parameters".

        :param str **key*: Parameter name.
        :return: The specified parameter.
        :rtype: ts.Tensor
        """
        assert isinstance(key, str), f"key has to be a {str} instance, not {type(key)}."
        return self.__getattribute__("__parameters")[key]
    
    @property
    def get_parameters(self) -> Dict[str, ts.Tensor]:
        """
        Property that creates a copy of parameters.

        :return: A copy of parameters.
        :rtype: Dict[str, ts.Tensor]
        """
        return copy.deepcopy(self.__getattribute__("__parameters"))
            
def _forward_unimplemented(self, *input: Any) -> None:
    raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forward" function.')

def _forward_with_specified_parameters_unimplemented(self, *input: Any) -> None:
    raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forward_with_specified_parameters" function.')

def _count_parameters_unimplemented(self) -> None:
    raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "count_parameters" function.')

def _reinitialize_unimplemented(self) -> None:
    raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "reinitialize" function.')

class Module:
    """
    Base module class.
    """
    has_parameters: bool
    _name: str
    def __init__(self, has_parameters: bool = True):
        """
        Initializes the Module class.

        :param bool, optional **has_parameters**: Boolean that indicates (and assigns) if the module has parameters. Set to `True`.
        """
        super().__setattr__("has_parameters", has_parameters)
        if self.has_parameters:
            super().__setattr__("parameters", Parameters())
            self.count_parameters= types.MethodType(_count_parameters_unimplemented, self)
            self.forward_with_specified_parameters = types.MethodType(_forward_with_specified_parameters_unimplemented, self)
            self.reinitialize = types.MethodType(_reinitialize_unimplemented, self)

        self.forward = types.MethodType(_forward_unimplemented, self)
        

    def __call__(self, *args, **kwds):
        """
        Built-in Python method that allows to call an instance like function.
        """
        return self.forward(*args, **kwds)

    @property
    def get_parameters(self) -> Dict[str, ts.Tensor]:
        """
        Property that returns module parameters.

        :return: Module parameters.
        :rtype: Dict[str, ts.Tensor]
        """
        return self.parameters.get_parameters
    
    def _define_name(self, name: str, **kwargs) -> None:
        """
        Method that defines the module name. Used by the __repr__ built-in Python method.

        :param str **name**: Module Name.
        :param ****kwargs**: Arbitrary keyword arguments used to complete the module name if necessary.
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
        Built-in Python method that represents the string representation of Module instance.

        :return: String representation of the module.
        :rtype: str
        """
        return self._name

class Activation_Module(Module):
    """
    Base activation module.
    """
    def __init__(self):
        """
        Initializes the Activation_Module class.
        """
        super().__init__(has_parameters=False)

    def _define_name(self, name: str) -> None:
        """
        Method that defines the activation module name. Used by the __repr__ built-in Python method.

        :param str **name**: Module Name.
        """
        self._name = name

class Sequential(Module):
    """
    Sequential is a modules container.
    """
    _modules: dict[str, Module]
    def __init__(self, *args):
        """
        Initializes the Sequential class.

        :param Tuple[Module] ***args**: Arguments containing modules.
        """
        super().__init__()
        super().__setattr__("_modules", {})
        if len(args) == 1:
            if isinstance(args[0], (list, tuple)):
                for enu, arg in enumerate(args[0]):
                    self.add_module(f"{enu}: {arg._name}", arg)
            else:
                for enu, arg in enumerate(args):
                    self.add_module(f"{enu}: {arg._name}", arg)
        else:
            for enu, arg in enumerate(args):
                self.add_module(f"{enu}: {arg._name}", arg)

        self.forward = self._forward
        self.forward_with_specified_parameters = self._forward_with_specified_parameters
        self.count_parameters = self._count_parameters
        self.reinitialize = self._reinitialize
        self.index = 0

    def __len__(self) -> int:
        """
        Built-in Python method that returns the length of Sequential.

        :return: Number of modules in the sequence.
        :rtype: int
        """
        return len(self._modules)
    
    def __iter__(self) -> Iterator:
        """
        Built-in Python method that returns an iterator.

        :return: Sequential Iterator.
        :rtype: Iterator
        """
        return iter(self._modules.items())
    
    def __getitem__(self, item: int) -> Module:
        """
        Get specified item according the index.

        :param int **item**: Module index.

        :return: Specified Module.
        :rtype: Module
        """
        assert isinstance(item, int), f"item has to be a {int} instance, not {type(item)}."

        return list(self._modules.items())[item][1]
        
    def __repr__(self) -> str:
        """
        Built-in Python method that represents the string representation of a Sequential instance.

        :return: String Sequential instance representation.
        :rtype: str
        """
        repr = "Sequential(\n"
        n = len(self._modules)
        for enu, name in enumerate(self._modules):
            repr += "    "
            repr += name
            if enu != n - 1:
                repr += "\n"
        repr += ")"
        return repr

    def _forward(self, input: ts.Tensor) -> ts.Tensor:
        """
        Method that forwards through the Sequential model.

        :param Tensor **input**: Input tensor.
        :return: Output Tensor.
        :rtype: Tensor
        """
        assert isinstance(input, ts.Tensor), f"input has to be a {ts.Tensor} instance, not {type(input)}."

        for _, module in self._modules.items():
            input = module(input)
        return input
    
    def _forward_with_specified_parameters(self, input: ts.Tensor, list_parameters: List[Parameters]):
        """
        Method that forwards through the Sequential model with specified parameters.

        :param Tensor **input**: Input tensor.
        :param List[Parameters] **list_parameters**: List containing modules parameters.
        :return: Output Tensor.
        :rtype: Tensor
        """
        assert isinstance(input, ts.Tensor), f"input has to be a {ts.Tensor} instance, not {type(input)}."

        i = 0
        for _, module in self._modules.items():
            if isinstance(module, Activation_Module):
                input = module.forward(input)
            else:
                input = module.forward_with_specified_parameters(input, list_parameters[i])
                i += 1
        return input

    def _ultimate_module_index(self) -> int:
        """
        Method that returns the index of the last module.

        :return: Last module index.
        :rtype: int
        """
        if len(self._modules) != 0:
            i = len(self._modules.items()) - 1
            modules = list(self._modules.items())
            while (i >= 0):
                if not isinstance(modules[i][1], Activation_Module):
                    pultimate = i
                    break
                i -= 1
            return pultimate
        else:
            raise TypeError(f"The instance is empty: {len()}")
        
    def forward_wu(self, input: ts.Tensor) -> ts.Tensor:
        """
        Method that forwards through the Sequential model without last module.

        :param Tensor **input**: Input tensor.
        :return: Output Tensor.
        :rtype: Tensor
        """
        assert isinstance(input, ts.Tensor), f"input has to be a {ts.Tensor} instance, not {type(input)}."
        for _, module in list(self._modules.items())[:self._ultimate_module_index()+1]:
            input = module(input)
        return input
    
    def forward_with_specified_parameters_wu(self, input: ts.Tensor, list_parameters: List[Parameters]):
        """
        Method that forwards through the Sequential model with specified parameters and without the last module.

        :param Tensor **input**: Input tensor.
        :param List[Parameters] **list_parameters**: List containing modules parameters.
        :return: Output Tensor.
        :rtype: Tensor
        """
        assert isinstance(input, ts.Tensor), f"input has to be a {ts.Tensor} instance, not {type(input)}."
        i = 0
        for _, module in list(self._modules.items())[:self._ultimate_module_index()+1]:
            if isinstance(module, Activation_Module):
                input = module.forward(input)
            else:
                input = module.forward_with_specified_parameters(input, list_parameters[i])
                i += 1
        return input
        
    def last_module_parameters(self) -> Parameters:
        """
        Method that returns the last module parameters.

        :return: Last module parameters.
        :rtype: Parameters
        """
        return list(self._modules.items())[self._ultimate_module_index()][1].parameters
            
    def add_module(self, name: str, module: Module) -> None:
        """
        Method that adds a new module into the sequence.
        """
        assert(name, str), f"name has to be a {str} instance, not {type(name)}."
        assert isinstance(module, Module), f"module has to be a {Module} instance, not {type(module)}."
        self._modules[name] = module

    def _count_parameters(self) -> int:
        """
        Method that returns the number of parameters contained in the sequence.

        :return: Total parameters contained in the modules.
        :rtype: int
        """
        return sum([module.count_parameters() for _, module in self._modules.items() if module.has_parameters])
    
    def _reinitialize(self) -> None:
        """
        Method that reinitializes all modules that have parameters.
        """
        for _, module in self._modules.items():
            if module.has_parameters:
                module.reinitialize()
    @property
    def get_parameters(self) -> List[Parameters]:
        """
        Property that returns the modules parameters contained in the sequence.

        :return: Modules parameters contained in the sequence.
        :rtype: List[Parameters]
        """
        return [module.get_parameters for _, module in self._modules.items() if module.has_parameters]