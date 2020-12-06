from typing import Any


def max_(*args) -> Any:
    """
    Because apparently I don't understand how it works so I have to prove it
    by recoding it...
    """
    if len(args) == 1:
        data_iter = iter(args[0])
    else:
        data_iter = iter(args)
    out = next(data_iter)
    for elem in data_iter:
        if elem > out:
            out = elem
    return out


def min_(*args) -> Any:
    """
    Because apparently I don't understand how it works so I have to prove it
    by recoding it...
    """
    if len(args) == 1:
        data_iter = iter(args[0])
    else:
        data_iter = iter(args)
    out = next(data_iter)
    for elem in data_iter:
        if elem < out:
            out = elem
    return out
