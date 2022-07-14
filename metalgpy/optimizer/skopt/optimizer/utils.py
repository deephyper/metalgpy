import metalgpy.optimizer.skopt as skopt
from ...._expression import Expression, Int, List, Float


def convert_to_skopt_dim(var_exp, surrogate_model=None):

    if surrogate_model in ["RF", "ET", "GBRT"]:
        # models not sensitive ot the metric space such as trees
        surrogate_model_type = "rule_based"
    else:
        # models sensitive to the metric space such as GP, neural networks
        surrogate_model_type = "distance_based"

    if isinstance(var_exp, Int):
        skopt_dim = skopt.space.Integer(
            low=var_exp._low,
            high=var_exp._high,
            # TODO: prior="log-uniform" if cs_hp.log else "uniform",
            name=var_exp.id,
        )
    elif isinstance(var_exp, Float):
        skopt_dim = skopt.space.Real(
            low=var_exp._low,
            high=var_exp._high,
            # TODO: prior="log-uniform" if cs_hp.log else "uniform",
            name=var_exp.id,
        )
    elif isinstance(var_exp, List):
        # the transform is important if we don't want the complexity of trees
        # to explode with categorical variables
        transform = "onehot" if surrogate_model_type == "distance_based" else "label"

        skopt_dim = skopt.space.Categorical(
            categories=list(range(len(var_exp._values))),
            name=var_exp.id,
            transform=transform,
        )
    else:
        raise TypeError(f"Cannot convert hyperparameter of type {type(var_exp)}")

    skopt_dim = skopt_dim if isinstance(skopt_dim, list) else [skopt_dim]
    return skopt_dim


def convert_to_skopt_space(exp, surrogate_model=None):
    """Convert an expression to a scikit-optimize Space."""

    # verify pre-conditions
    if not (isinstance(exp, Expression)):
        raise TypeError("Input should be of type Expression")

    # convert the Expression to skopt.space.Space
    dimensions = []
    for var_exp in exp.variables().values():
        dimensions.extend(convert_to_skopt_dim(var_exp, surrogate_model))

    return dimensions
