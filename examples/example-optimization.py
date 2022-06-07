import metalgpy as mpy
import numpy as np
import tree


@mpy.meta
def f(x):
    return x


@mpy.meta
def g(x):
    return -x


@mpy.meta
def h(x):
    return x + 1

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

        if isinstance(var_exp, mpy.List) and isinstance(var_exp[s[-1]], mpy.Expression):
            d = sample_from_var(var_exp[s[-1]].choice(), rng)
            s.extend(d)
        
    return s

def query_all_variables(choices: list) -> dict:

    s = []  # var
    for var_exp in choices:

        s.append(var_exp)

        if isinstance(var_exp, mpy.List):
            sub_choices = var_exp.sub_choice()
            sub_variables = query_all_variables(sub_choices)
            if len(sub_variables) > 0:
                s.append(sub_variables)
    return s

def sample_all_variables(choices, rng):

    print("choices: ", choices)
    s = []  # var
    for var_exp in choices:

        s.append(var_exp.sample(rng=rng))

        if isinstance(var_exp, mpy.List):
            sub_choices = var_exp.choice()
            sub_variables = sample_all_variables(sub_choices, rng)
            if len(sub_variables) > 0:
                s.append(sub_variables)

    return s

def filter_nested_structure(l):

    def filter_nested_aux(l, parent=None):
        
        if type(l) is list:
            head, *tail = l
        
            if type(head) is list:
                if len(tail) == 0:
                    return [*filter_nested_aux(head[parent])]
                else:
                    return [*filter_nested_aux(head[parent]), *filter_nested_aux(tail)]
            else:
                if len(tail) == 0:
                    return [head]
                else:
                    return [head, *filter_nested_aux(tail, parent=head)]
        
        else:
            return [l]
        
    l = filter_nested_aux(l)

    return l


if __name__ == "__main__":
    rng = np.random.RandomState(46)

    # program = h(mpy.List([f(mpy.List([1, 3, 5])), g(mpy.List([2, 4, 6]))]))

    # choices = program.choice()

    # variables_structure = query_all_variables(choices)
    # print(variables_structure)

    # variables_names= tree.map_structure(lambda x: x.var_id, variables_structure)

    # print(variables_names)

    # s = sample_all_variables(choices, rng)
    # print("sample: ", s)

    # print(tree.flatten(variable_names))

    # print(tree.unflatten_as(variable_names, [2,0]))

    # s = [0, 2, 0] # [0, [2, 0]]
    # s_struc = tree.unflatten_as(variables_structure, s)
    # print(s_struc)

    # s_struc = [0, [0, 1], 1, [0, 1]]
    # res = filter_nested_structure(s_struc)
    # print(res)

    mpy.VarExpression.var_id = 0

    f_ = f(mpy.List([
        -1,
        mpy.List(["a",mpy.List([0,1,2]),"b"]),
        mpy.Int(10, 20),
    ]))

    sample_new = lambda : tree.map_structure(lambda x: x.sample(rng=rng), query_all_variables(f_.choice()))

    # sample 1
    s = sample_new()
    print("sample: ", s)
    s = filter_nested_structure(s)
    f_clone = f_.clone()
    f_clone.freeze(s)
    print(f"f_clone.freeze({s}): ", f_clone)
    res = f_clone.evaluate()
    print(f"{f_clone} =", res, end="\n\n")

    # sample 2
    s = sample_new()
    print("sample: ", s)
    s = filter_nested_structure(s)
    f_clone = f_.clone()
    f_clone.freeze(s)
    print(f"f_clone.freeze({s}): ", f_clone)
    res = f_clone.evaluate()
    print(f"{f_clone} =", res, end="\n\n")

    # sample 3
    s = sample_new()
    print("sample: ", s)
    s = filter_nested_structure(s)
    f_clone = f_.clone()
    f_clone.freeze(s)
    print(f"f_clone.freeze({s}): ", f_clone)
    res = f_clone.evaluate()
    print(f"{f_clone} =", res, end="\n\n")



