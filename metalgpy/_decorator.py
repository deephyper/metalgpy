from ._expression import ObjectExpression


def meta(obj):
    """Transform an object into an ``ObjectExpression`` Object."""

    cls_attrs = {}
    meta_class = type(
        f"ObjectExpression_{obj.__name__}", (ObjectExpression,), cls_attrs
    )
    meta_obj = meta_class(obj)

    return meta_obj