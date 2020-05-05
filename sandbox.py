import numpy as np

t = np.array([
    [True, True],
    [True, False],
    [False, True],
    [False, False],
    [True, False]
],dtype=np.bool)


# print(t)


def boolean_search(query, ds, mode = "and"):
    if mode == "and":
        res = np.ones(shape=ds.shape[0])
    else:
        res = np.zeros(shape=ds.shape[0])

    for (idx, val) in query:
        indices = ds[:, idx] == val
        if mode == "and":
            res = np.logical_and(res, indices)
        else:
            res = np.logical_or(res, indices)
    return res