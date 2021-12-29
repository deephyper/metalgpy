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

    def __call__(self, *args, **kwargs):
        return ExpressionCallExpression(self, *args, **kwargs)

    def choice(self):
        choices = []

        def choice_aux(o):

            if isinstance(o, VarExpression):
                choices.append(o)

            if isinstance(o, Expression):
                choices.extend(o.choice())

            return choices

        tree.map_structure(choice_aux, self.__dict__)

        return choices[::-1]

    def freeze(self, choice):

        Q = (
            choice
            if isinstance(choice, collections.deque)
            else collections.deque(choice[::-1])
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

        self.map(evaluate_aux)

    @abc.abstractmethod
    def evaluate(self):
        pass

    def map(self, function):
        self.__dict__ = tree.map_structure(function, self.__dict__)

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

    def clone(self):
        c = copy.deepcopy(self)
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

    def __getattribute__(self, __name: str):
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            return ExpressionAttributeAccess(self, __name)

    def evaluate(self):

        res = self._evaluate()
        while isinstance(res, Expression):
            res = res.evaluate()
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


class ExpressionAttributeAccess(Expression):
    def __init__(self, expression, name) -> None:
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
        elif inspect.isfunction(self._obj):
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

    def __eq__(self, other):
        return type(self) is type(other) and self.value == other.value

    @abc.abstractmethod
    def sample(self, size=None, rng=None):
        raise NotImplementedError

    def freeze(self, choice):
        self.value = choice.popleft()

    def evaluate(self):
        return self.value


class List(VarExpression):
    def __init__(self, l) -> None:
        self._array = list(l)

    def __repr__(self) -> str:
        if not(self.value is None):
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
        idx = choice.popleft()
        if idx < 0 or idx >= len(self):
            raise ValueError(
                f"choice for variable List should be a correct index but is {idx}"
            )
        self.value = self._array[idx]

        if isinstance(self.value, Expression):
            self.value.freeze(choice)

    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random.RandomState()

        idx = rng.choice(len(self), size=size)
        return idx


class Int(VarExpression):
    def __init__(self, lower, upper) -> None:
        self._lower = lower
        self._upper = upper

    def __repr__(self) -> str:
        if not(self.value is None):
            return self.value.__repr__()
        else:
            return f"Int({self._lower}, {self._upper})"

    def __eq__(self, other):
        b = super().__eq__(other)
        return b and self._upper == other._upper and self._lower == other._lower

    def sample(self, size=None, rng=None):

        if rng is None:
            rng = np.random.RandomState()

        return rng.randint(self._lower, self._upper, size=size)


def sample(n, exp):
    for i in range(n):
        yield exp.clone()
