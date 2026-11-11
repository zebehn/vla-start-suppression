"""Paired tests, intervals and quantiles used by the scorer (standard library only).

Conventions frozen in spec v0:
- thresholds compare integer counts: at_least(k, n, num, den) is k >= n * num // den, never a float ratio;
- the median is statistics.median and other quantiles are linear interpolation (statistics.quantiles, "inclusive");
- Wilson intervals use z = 1.959963984540054 (two-sided 95%).
"""
import math
import statistics

Z95 = 1.959963984540054


def mcnemar_exact(a, b):
    """Two-sided exact McNemar test on paired 0/1 lists. Returns (p, b01, b10), b01 = #(a=0, b=1), b10 = #(a=1, b=0)."""
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError("paired lists differ in length")
    b01 = sum(1 for x, y in zip(a, b) if not x and y)
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = b01 + b10
    if n == 0:
        return 1.0, b01, b10
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p), b01, b10


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact test for [[a, b], [c, d]] (sum of tables no more probable than the observed one)."""
    row, col, n = a + b, a + c, a + b + c + d

    def p(x):
        return math.comb(col, x) * math.comb(n - col, row - x) / math.comb(n, row)
    lo, hi = max(0, row + col - n), min(row, col)
    p0 = p(a)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9)))


def wilson(k, n, z=Z95):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def at_least(k, n, num, den=100):
    """Integer-count threshold: k >= floor(n * num / den). Example: 82.5% of 200 is at_least(k, 200, 825, 1000)."""
    return k >= n * num // den


def below(k, n, num, den=100):
    return k < n * num // den


def median(xs):
    return statistics.median(xs)


def quantile(xs, i, n):
    """i-th n-quantile (e.g. i=1, n=10 is the 10th percentile) by linear interpolation.

    Uses statistics.quantiles(method="inclusive"), which equals numpy's default up to float rounding; the gate
    threshold tau keeps numpy.percentile (gate.calibrate) because the committed thresholds were computed with it.
    """
    s = sorted(xs)
    if not s:
        raise ValueError("empty")
    if len(s) == 1:
        return s[0]
    return statistics.quantiles(s, n=n, method="inclusive")[i - 1]


def five_number(xs):
    """min, p25, median, p75, max with statistics.quantiles(method='inclusive') for the quartiles."""
    s = sorted(x for x in xs if x is not None)
    if not s:
        return None
    if len(s) == 1:
        return {"min": s[0], "p25": s[0], "median": s[0], "p75": s[0], "max": s[0]}
    q1, _, q3 = statistics.quantiles(s, n=4, method="inclusive")
    return {"min": s[0], "p25": q1, "median": statistics.median(s), "p75": q3, "max": s[-1]}


def entropy_normalized(counts, base):
    """Shannon entropy of a count vector divided by log(base); None if the total is 0."""
    n = sum(counts)
    if n == 0:
        return None
    h = -sum((c / n) * math.log(c / n) for c in counts if c > 0)
    return h / math.log(base)


def tv_distance(p, q, cats):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in cats)
