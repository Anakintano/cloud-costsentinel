"""Six Sigma control-chart math: I-MR limits, Nelson rules, capability vs budget."""
import numpy as np

D2 = 1.128  # d2 constant for moving range of size 2


def imr(x):
    """Individuals chart: center, sigma-hat (MR-bar/d2), UCL, LCL."""
    x = np.asarray(x, float)
    mr_bar = np.abs(np.diff(x)).mean()
    sigma = mr_bar / D2
    c = x.mean()
    return {"center": c, "sigma": sigma, "ucl": c + 3 * sigma, "lcl": c - 3 * sigma}


def nelson(x, center, sigma):
    """Return {rule: [indices]}. Rules: 1 beyond 3σ, 2 nine same side, 3 six trending."""
    x = np.asarray(x, float)
    hits = {1: [], 2: [], 3: []}
    hits[1] = [int(i) for i in np.where(np.abs(x - center) > 3 * sigma)[0]]
    side = np.sign(x - center)
    run = 0
    for i in range(len(x)):
        run = run + 1 if i and side[i] == side[i - 1] and side[i] != 0 else 1
        if run >= 9:
            hits[2].append(i)
    d = np.sign(np.diff(x))
    run = 0
    for i in range(1, len(x)):
        run = run + 1 if i > 1 and d[i - 1] == d[i - 2] and d[i - 1] != 0 else 1
        if run >= 5:  # 5 same-direction steps = 6 points
            hits[3].append(i)
    return hits


def capability(x, usl):
    """Upper-spec-only capability (cost has no lower limit): Cpu == Cpk. >1.33 healthy, <1 over budget."""
    x = np.asarray(x, float)
    s = imr(x)
    cpu = (usl - s["center"]) / (3 * s["sigma"]) if s["sigma"] else float("inf")
    return {"cpu": cpu, "cpk": cpu}
