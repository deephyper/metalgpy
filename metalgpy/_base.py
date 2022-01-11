import abc
import collections
import copy
import inspect

import numpy as np
import tree


# https://docs.python.org/3/reference/datamodel.html#special-method-names
# https://docs.python.org/3/library/operator.html
BIN_OPS_SYNTAX_2_ATTR = {
    # comparison
    "<": "__lt__",
    "<=": "__le__",
    "==": "__eq__",
    "!=": "__ne__",
    ">": "__gt__",
    ">=": "__ge__",
    # numeric operators
    "+": "__add__",
    "-": "__sub__",
    "*": "__mul__",
    "@": "__matmul__",
    "/": "__truediv__",
    "//": "__floordiv__",
    "%": "__mod__",
    # __divmod__
    "**": "__pow__",
    # __lshift__
    # __rshift__
    "and": "__and__",
    "^": "__xor__",
    "or": "__or__",
}
BIN_OPS_ATTR_2_SYNTAX = {v: k for k, v in BIN_OPS_SYNTAX_2_ATTR.items()}

# unary operators
UNA_OPS_SYNTAX_2_ATTR = {
    "-": "__neg__",
    "+": "__pos__",
    "~": "__invert__",
}
UNA_OPS_ATTR_2_SYNTAX = {v: k for k, v in UNA_OPS_SYNTAX_2_ATTR.items()}


class Expression:

    # numerical operators
    def __add__(self, other):
        return BinaryExpression(self, other, BIN_OPS_ATTR_2_SYNTAX["__add__"])

    def __sub__(self, other):
        return BinaryExpression(self, other, BIN_OPS_SYNTAX_2_ATTR["__sub__"])

    def __mul__(self, other):
        return BinaryExpression(self, other, BIN_OPS_SYNTAX_2_ATTR["__mul__"])

    def __matmul__(self, other):
        return BinaryExpression(self, other, BIN_OPS_SYNTAX_2_ATTR["__matmul__"])

    def __truediv__(self, other):
        return BinaryExpression(self, other, BIN_OPS_SYNTAX_2_ATTR["__truediv__"])

    def __floordiv__(self, other):
        return BinaryExpression(self, other, BIN_OPS_SYNTAX_2_ATTR["__floordiv__"])

    def __mod__(self, other):
        return BinaryExpression(self, other, BIN_OPS_SYNTAX_2_ATTR["__mod__"])

    # order and comparison operators
    def __lt__(self, other):
        return BinaryExpression(self, other, BIN_OPS_SYNTAX_2_ATTR["__lt__"])

    def __le__(self, other):
        return BinaryExpression(self, other, BIN_OPS_SYNTAX_2_ATTR["__le__"])

    def __eq__(self, other):
        return BinaryExpression(self, other, BIN_OPS_SYNTAX_2_ATTR["__eq__"])

    def __ne__(self, other):
        return BinaryExpression(self, other, BIN_OPS_SYNTAX_2_ATTR["__ne__"])

    def __gt__(self, other):
        return BinaryExpression(self, other, BIN_OPS_SYNTAX_2_ATTR["__gt__"])

    def __ge__(self, other):
        return BinaryExpression(self, other, BIN_OPS_SYNTAX_2_ATTR["__ge__"])

    # logic operators
    def __and__(self, other):
        return BinaryExpression(self, other, BIN_OPS_SYNTAX_2_ATTR["__and__"])

    def __xor__(self, other):
        return BinaryExpression(self, other, BIN_OPS_SYNTAX_2_ATTR["__xor__"])

    def __or__(self, other):
        return BinaryExpression(self, other, BIN_OPS_SYNTAX_2_ATTR["__or__"])

    # unary operators
    def __pos__(self):
        return UnaryExpression(UNA_OPS_SYNTAX_2_ATTR["__pos__"], self)

    def __neg__(self):
        return UnaryExpression(UNA_OPS_SYNTAX_2_ATTR["__neg__"], self)

    def __invert__(self):
        return UnaryExpression(UNA_OPS_SYNTAX_2_ATTR["__invert__"], self)

    def __getitem__(self, item):
        return ExpressionItemAccess(self, item)

    def __getattribute__(self, __name: str):
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            return ExpressionAttributeAccess(self, __name)

    def __call__(self, *args, **kwargs):
        return ExpressionCallExpression(self, *args, **kwargs)

    def choice(self):
        choices = []

        def choice_aux(o):

            if isinstance(o, Expression):

                if isinstance(o, VarExpression):
                    choices.append(o)
                else:
                    choices.append(o.choice())

            return choices

        tree.map_structure(choice_aux, self.__dict__)

        choices = tree.flatten(choices)

        return choices

    def freeze(self, choice):

        Q = (
            choice
            if isinstance(choice, collections.deque)
            else collections.deque(choice)
        )

        # propagate the materialization
        def freeze_aux(o):

            if isinstance(o, Expression):
                o.freeze(Q)

        tree.map_structure(freeze_aux, self.__dict__)

    def evaluate_children(self):

        # propagate the evaluation
        def evaluate_aux(o):

            if isinstance(o, Expression):
                eval_ = o.evaluate()
            else:
                eval_ = o
            return eval_

        self.__dict__ = tree.map_structure(evaluate_aux, self.__dict__)

    @abc.abstractmethod
    def evaluate(self):
        pass

    def __copy__(self):
        cls = self.__class__
        obj = cls.__new__(cls)
        obj.__dict__.update(copy.copy(self.__dict__))
        return obj

    def __deepcopy__(self, memo):
        cls = self.__class__
        obj = cls.__new__(cls)
        obj.__dict__.update(copy.deepcopy(self.__dict__))
        return obj

    def clone(self, deep=False):
        if deep:
            c = copy.deepcopy(self)
        else:
            c = copy.copy(self)
        return c


class BinaryExpression(Expression):
    """left 'operation' right"""

    def __init__(self, left, right, operator):
        self.left = left
        self.right = right
        self.operator = operator

    def __repr__(self) -> str:
        return f"{self.left} {self.operator} {self.right}"

    def evaluate(self):

        self.evaluate_children()

        operator_implementation = getattr(
            self.left, BIN_OPS_SYNTAX_2_ATTR[self.operator]
        )

        return operator_implementation(self.right)


class UnaryExpression(Expression):
    """operation x"""

    def __init__(self, operator, x):
        self.operator = operator
        self.x = x

    def __repr__(self) -> str:
        return f"{self.operator}{self.x}"

    def evaluate(self):
        self.evaluate_children()

        operator_implementation = getattr(self.x, UNA_OPS_SYNTAX_2_ATTR[self.operator])

        return operator_implementation()


class FunctionCallExpression(Expression):
    def __init__(self, function_parent, function, *args, **kwargs) -> None:
        self.function_parent = function_parent
        self.function = function
        self.args = list(args)
        self.kwargs = kwargs

    def __repr__(self) -> str:
        args = str(self.args)[1:-1]

        kwargs = ""
        for i, (k, v) in enumerate(self.kwargs.items()):
            kwargs += f"{k}={v}"
            if i < len(self.kwargs) - 1:
                kwargs += ", "

        if args != "" and kwargs != "":
            args += ", "

        prefix = self.function_parent.__name__ if self.function_parent else ""
        fname = self.function.__name__
        if fname == "__call__" or fname == "__init__":
            fname = ""

        return (
            f"{prefix}{'.' if prefix and fname != '' else ''}{fname}({args + kwargs})"
        )

    def evaluate(self):

        res = self._evaluate()
        # while isinstance(res, Expression):
        #     res = res.evaluate()
        return res

    def _evaluate(self):

        self.evaluate_children()

        if self.function_parent:
            if self.function.__name__ == "__init__":
                return self.function_parent(*self.args, **self.kwargs)
            else:
                return self.function(self.function_parent, *self.args, **self.kwargs)
        else:
            return self.function(*self.args, **self.kwargs)


class ExpressionItemAccess(Expression):
    def __init__(self, expression, item):
        self.expression = expression
        self.item = item

    def __repr__(self) -> str:
        return f"{self.expression}[{self.item}]"

    def evaluate(self):
        self.evaluate_children()

        return self.expression[self.item]


class ExpressionAttributeAccess(Expression):
    def __init__(self, expression, name):
        self.expression = expression
        self.name = name

    def __repr__(self) -> str:
        return f"{self.expression}.{self.name}"

    def evaluate(self):
        self.evaluate_children()

        return getattr(self.expression, self.name)


class ExpressionCallExpression(Expression):
    def __init__(self, expression, *args, **kwargs) -> None:
        self.expression = expression
        self.args = list(args)
        self.kwargs = kwargs

    def __repr__(self) -> str:
        args = str(self.args)[1:-1]

        kwargs = ""
        for i, (k, v) in enumerate(self.kwargs.items()):
            kwargs += f"{k}={v}"
            if i < len(self.kwargs) - 1:
                kwargs += ", "

        if args != "" and kwargs != "":
            args += ", "

        return f"{self.expression}({args + kwargs})"
    
    def choice(self):
        choices = self.expression.choice()

        def choice_aux(o):

            if isinstance(o, Expression):

                if isinstance(o, VarExpression):
                    choices.append(o)
                else:
                    choices.append(o.choice())

            return choices

        tree.map_structure(choice_aux, self.args)
        tree.map_structure(choice_aux, self.kwargs)

        choices = tree.flatten(choices)

        return choices

    def freeze(self, choice):

        Q = (
            choice
            if isinstance(choice, collections.deque)
            else collections.deque(choice)
        )
        self.expression.freeze(Q)

        # propagate the materialization
        def freeze_aux(o):
            
            if isinstance(o, Expression):
                o.freeze(Q)

        tree.map_structure(freeze_aux, self.args)
        tree.map_structure(freeze_aux, self.kwargs)

    def evaluate(self):

        self.evaluate_children()

        return self.expression(*self.args, **self.kwargs)


class ObjectExpression(Expression):
    def __init__(self, obj):
        self._obj = obj

    def __call__(self, *args, **kwargs):

        if inspect.isclass(self._obj):
            return FunctionCallExpression(
                self._obj,
                self._obj.__init__,
                *args,
                **kwargs,
            )
        elif inspect.isfunction(self._obj) or (inspect.isbuiltin(self._obj) and not(inspect.ismethod(self._obj))):
            return FunctionCallExpression(
                None,  # a function does not have a parent (stateless)
                self._obj,
                *args,
                **kwargs,
            )
        else:
            return FunctionCallExpression(
                self._obj,
                self._obj.__call__,
                *args,
                **kwargs,
            )

    def __repr__(self):
        return self._obj.__repr__()

    def evaluate(self):
        self.evaluate_children()

        return self._obj


def meta(obj):
    """Transform an object into a ObjectExpression Object."""

    cls_attrs = {}
    meta_class = type(
        f"ObjectExpression_{obj.__name__}", (ObjectExpression,), cls_attrs
    )
    meta_obj = meta_class(obj)

    return meta_obj


class VarExpression(Expression):

    value = None
    var_id = 0

    def __init__(self) -> None:
        super().__init__()
        self.var_id = VarExpression.var_id
        VarExpression.var_id += 1

    def __eq__(self, other):
        return type(self) is type(other) and self.value == other.value

    @abc.abstractmethod
    def sample(self, size=None, rng=None):
        raise NotImplementedError

    @property
    def id(self):
        return self.var_id

    def freeze(self, choice):
        self.value = choice.popleft()

    def evaluate(self):
        if isinstance(self.value, Expression):
            return self.value.evaluate()
        else:
            return self.value
    
    def choice(self):
        return [self]


class List(VarExpression):
    def __init__(self, l):
        super().__init__()
        self._array = list(l)

    def __repr__(self) -> str:
        if not (self.value is None):
            return self.value.__repr__()
        else:
            str_ = str(self._array)[1:-1]
            return f"List({str_})"

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._array[item]
        elif isinstance(item, collections.abc.Iterable):
            return [self._array[i] for i in item]
        else:
            raise ValueError("index of List should be int or iterable!")

    def __len__(self):
        return len(self._array)

    def __eq__(self, other):
        b = super().__eq__(other)
        return b and self._array == other._array

    def freeze(self, choice):

        # convert to queue if not already done
        choice = (
            choice
            if isinstance(choice, collections.deque)
            else collections.deque(choice)
        )

        idx = choice.popleft()
        if idx < 0 or idx >= len(self):
            raise ValueError(
                f"choice for variable {self} should be a correct index but is {idx}"
            )
        self.value = self._array[idx]

        if isinstance(self.value, Expression):
            self.value.freeze(choice)

    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random.RandomState()

        idx = rng.choice(len(self), size=size)
        return idx

    def sub_choice(self):
        return Expression.choice(self)[::-1]

class Int(VarExpression):
    """Defines an discrete variable.

        Args:
            lower (int): the lower bound of the variable discrete interval.
            upper (int): the upper bound of the variable discrete interval.
    """

    def __init__(self, lower: int, upper: int):
        super().__init__()
        self._lower = lower
        self._upper = upper

    def __repr__(self) -> str:
        if not (self.value is None):
            return self.value.__repr__()
        else:
            return f"Int({self._lower}, {self._upper})"

    def __eq__(self, other):
        b = super().__eq__(other)
        return b and self._upper == other._upper and self._lower == other._lower

    def freeze(self, choice):

        # convert to queue if not already done
        choice = (
            choice
            if isinstance(choice, collections.deque)
            else collections.deque(choice)
        )

        choice = choice.popleft()
        if self._lower > choice or choice > self._upper:
            raise ValueError(
                f"choice for variable {self} should be between [{self._lower}, {self._upper}] but is {choice}"
            )
        self.value = choice

    def sample(self, size=None, rng=None):

        if rng is None:
            rng = np.random.RandomState()

        return rng.randint(self._lower, self._upper, size=size)



class Float(VarExpression):
    """Defines a continuous variable.

        Args:
            lower (float): the lower bound of the variable continuous interval.
            upper (float): the upper bound of the variable continuous interval.
    """

    def __init__(self, lower: float, upper: float) -> None:
        
        super().__init__()
        self._lower = lower
        self._upper = upper

    def __repr__(self) -> str:
        if not (self.value is None):
            return self.value.__repr__()
        else:
            return f"Float({self._lower}, {self._upper})"

    def __eq__(self, other):
        b = super().__eq__(other)
        return b and self._upper == other._upper and self._lower == other._lower
    
    def freeze(self, choice):

        # convert to queue if not already done
        choice = (
            choice
            if isinstance(choice, collections.deque)
            else collections.deque(choice)
        )

        choice = choice.popleft()
        if self._lower > choice or choice > self._upper:
            raise ValueError(
                f"choice for variable {self} should be between [{self._lower}, {self._upper}] but is {choice}"
            )
        self.value = choice

    def sample(self, size=None, rng=None):

        if rng is None:
            rng = np.random.RandomState()

        return rng.uniform(self._lower, self._upper, size=size)


# class LazyVar(VarExpression):
    
#     def __init__(self, var_exp, function) -> None:
#         super().__init__()
#         self.var_exp = var_exp
#         self.function = function

#     def freeze(self, choice):

#         # convert to queue if not already done
#         choice = (
#             choice
#             if isinstance(choice, collections.deque)
#             else collections.deque(choice)
#         )

#         idx = choice.popleft()
#         if idx < 0 or idx >= len(self):
#             raise ValueError(
#                 f"choice for variable {self} should be a correct index but is {idx}"
#             )
#         self.value = self._array[idx]

#         if isinstance(self.value, Expression):
#             self.value.freeze(choice)
        



# class Repeat(Expression):
#     def __init__(self, n, function=None):
#         super().__init__()
#         self.n = n
#         self.function = function
#         self._repeat = None

#     def __repr__(self) -> str:
#         if not (self._repeat is None):
#             return self._repeat.__repr__()
#         else:
#             return f"Repeat({self.n}, {self.function})"

#     def __getitem__(self, item):
#         if self._repeat is not None:
#             if isinstance(item, int):
#                 return self._repeat[item]
#             elif isinstance(item, collections.abc.Iterable):
#                 return [self._repeat[i] for i in item]
#             else:
#                 raise IndexError("index of Repeat should be int or iterable!")
#         else:
#             raise IndexError(f"index of {self} was accessed without being frozen first!")

#     def __len__(self):
#         if isinstance(self.n, Expression):
#             return self.n.value
#         else:
#             return self.n

#     def freeze(self, choice):

#         # convert to queue if not already done
#         choice = (
#             choice
#             if isinstance(choice, collections.deque)
#             else collections.deque(choice)
#         )

#         idx = choice.popleft()
#         if idx < 0 or idx >= len(self):
#             raise ValueError(
#                 f"choice for variable {self} should be a correct index but is {idx}"
#             )
#         self.value = self._array[idx]

#         if isinstance(self.value, Expression):
#             self.value.freeze(choice)

# sample programs


def sample_from_var(choices: list, rng) -> dict:
    """Sample variables from a list of variable choice.

    Args:
        choices (list): List of variables to sample.
        rng (np.random.RandomState): the random state.

    Returns:
        dict: keys are variable ids, values are variables values.
    """
    s = [] # var samples
    for var_exp in choices:

        s.append(var_exp.sample(rng=rng))

        if isinstance(var_exp, List) and isinstance(var_exp[s[-1]], Expression):
            d = sample_from_var(var_exp[s[-1]].choice(), rng)
            s.extend(d)
        
    return s


def sample_choices(exp, size=1, rng=None):

    if rng is None:
        rng = np.random.RandomState()

    choices = exp.choice()
    for _ in range(size):

        variable_choice = sample_from_var(choices, rng)
        yield variable_choice


def sample_programs(exp, size, rng=None, deep=False):

    if rng is None:
        rng = np.random.RandomState()

    for variable_choice in sample_choices(exp, size, rng):
        exp_clone = exp.clone(deep=deep)
        exp_clone.freeze(variable_choice)

        yield variable_choice, exp_clone
