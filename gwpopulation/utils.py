import os

from .cupy_utils import erf, gammaln, xp


def beta_dist(xx, alpha, beta, scale=1):
    if alpha < 0:
        raise ValueError(f"Parameter alpha must be greater or equal zero, low={alpha}.")
    if beta < 0:
        raise ValueError(f"Parameter beta must be greater or equal zero, low={beta}.")
    ln_beta = (alpha - 1) * xp.log(xx) + (beta - 1) * xp.log(scale - xx)
    ln_beta -= betaln(alpha, beta)
    ln_beta -= (alpha + beta - 1) * xp.log(scale)
    prob = xp.exp(ln_beta)
    prob = xp.nan_to_num(prob)
    prob *= (xx >= 0) * (xx <= scale)
    return prob


def betaln(alpha, beta):
    ln_beta = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
    return ln_beta


def powerlaw(xx, alpha, high, low):
    if xp.any(xp.asarray(low) < 0):
        raise ValueError(f"Parameter low must be greater or equal zero, low={low}.")
    if alpha == -1:
        norm = 1 / xp.log(high / low)
    else:
        norm = (1 + alpha) / (high ** (1 + alpha) - low ** (1 + alpha))
    prob = xp.power(xx, alpha)
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob


def truncnorm(xx, mu, sigma, high, low):
    if sigma <= 0:
        raise ValueError(f"Sigma must be greater than 0, sigma={sigma}")
    norm = 2 ** 0.5 / xp.pi ** 0.5 / sigma
    norm /= erf((high - mu) / 2 ** 0.5 / sigma) + erf((mu - low) / 2 ** 0.5 / sigma)
    prob = xp.exp(-xp.power(xx - mu, 2) / (2 * sigma ** 2))
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob

def farrow_primary(xx, kappa, mur1, sigmar1, mur2, sigmar2, mus, sigmas, high, low):
    if sigmar1 <= 0:
        raise ValueError(f"Sigmar1 must be greater than 0, sigmar1={sigmar1}")
    if sigmar2 <= 0:
        raise ValueError(f"Sigmar2 must be greater than 0, sigmar2={sigmar2}")
    if sigmas <= 0:
        raise ValueError(f"Sigmas must be greater than 0, sigmas={sigmas}")
    prob = xp.exp(-xp.power(xx - mus, 2) / (2 * sigmas ** 2)) / (2 * xp.power(2 * xp.pi, 0.5) * sigmas) * (kappa * (erf((xx - mur1) / (2 ** 0.5 * sigmar1)) + erf(mur1 / (2 ** 0.5 * sigmar1))) + (1 - kappa) * (erf((xx - mur2) / (2 ** 0.5 * sigmar2)) + erf(mur2 / (2 ** 0.5 * sigmar2)))) + 0.25 * xp.power(2 / xp.pi, 0.5) * (kappa * xp.exp(-xp.power(xx - mur1, 2) / (2 * sigmar1 ** 2)) / sigmar1 + (1 - kappa) * xp.exp(-xp.power(xx - mur2, 2) / (2 * sigmar2 ** 2)) / sigmar2) * (erf((xx - mus) / (2 ** 0.5 * sigmas)) + erf(mus / (2 ** 0.5 * sigmas)))
    norm = sum(xx * prob)
    prob /= norm
    prob *= (xx <= high) & (xx >= low)
    return prob

def farrow_secondary(xx, kappa, mur1, sigmar1, mur2, sigmar2, mus, sigmas, high, low):
    if sigmar1 <= 0:
        raise ValueError(f"Sigmar1 must be greater than 0, sigmar1={sigmar1}")
    if sigmar2 <= 0:
        raise ValueError(f"Sigmar2 must be greater than 0, sigmar2={sigmar2}")
    if sigmas <= 0:
        raise ValueError(f"Sigmas must be greater than 0, sigmas={sigmas}")
    prob = xp.exp(-xp.power(xx - mus, 2) / (2 * sigmas **2)) / (2 * xp.power(2 * xp.pi, 0.5) * sigmas) * (kappa * (1 - erf((xx - mur1) / (2 ** 0.5 * sigmar1))) + (1 - kappa) * (1 - erf((xx - mur2) / (2 ** 0.5 * simgar2)))) + 0.25 * xp.power(2 / xp.pi, 0.5) * (kappa * xp.exp(-xp.power(xx - mur1, 2) / (2 *sigmar1 **2)) / sigmar1 + (1 - kappa) * xp.exp(-xp.power(xx - mur2, 2) / (2 *sigmar2 **2)) / sigmar2) * (1 - erf((xx - mus) / (2 ** 0.5 * sigmas)))
    norm = sum(xx * prob)
    prob /= norm
    prob *= (xx<= high) & (xx >= low)
    return prob

def farrow_slow_peak_primary(xx, kappar, mur1, sigmar1, mur2, sigmar2, kappas, mus1, sigmas1, mus2, sigmas2, high, low):
    if sigmar1 <= 0:
        raise ValueError(f"Sigmar1 must be greater than 0, sigmar1={sigmar1}")
    if sigmar2 <= 0:
        raise ValueError(f"Sigmar2 must be greater than 0, sigmar2={sigmar2}")
    if sigmas1 <= 0:
        raise ValueError(f"Sigmas1 must be greater than 0, sigmas1={sigmas1}")
    if sigmas2 <= 0:
        raise ValueError(f"Sigmas2 must be greater than 0, sigmas2={sigmas2}")
    prob = 0.25 * (xp.power(2 / xp.pi, 0.5) * (kappas * xp.exp(-xp.power(xx - mus1, 2) / (2 * sigmas1 **2)) / sigmas1 + (1 - kappas) * xp.exp(-xp.power(xx - mus2, 2) / (2 * sigmas2 **2)) / sigmas2) * (kappar * (erf((xx - mur1) / (2 ** 0.5 * sigmar1)) + erf(mur1 / (2 ** 0.5 *sigmar1))) + (1 - kappar) * (erf((xx - mur2) / (2 ** 0.5 * sigmar2)) + erf(mur2 / (2 ** 0.5 * sigmar2)))) + xp.power(2 / xp.pi, 0.5) * (kappar * xp.exp(-xp.power(xx - mur1, 2) / (2 * sigmar1 **2)) / sigmar1 + (1 - kappar) * xp.exp(-xp.power(xx - mur2, 2) / (2 * sigmar2 ** 2)) / sigmar2) * (kappas * (erf((xx - mus1) / (2 ** 0.5 * sigmas1)) + erf(mus1 / (2 ** 0.5 * sigmas1))) + (1 - kappas) * (erf((xx - mus2) / (2 ** 0.5 * sigmas2)) + erf(mus2 / (2 ** 0.5 * sigmas2)))))
    norm = sum(xx * prob)
    prob /= norm
    prob *= (xx<= high) & (xx >= low)
    return prob

def farrow_slow_peak_secondary(xx, kappar, mur1, sigmar1, mur2, sigmar2, kappas, mus1, sigmas1, mus2, sigmas2, high, low):
    if sigmar1 <= 0:
        raise ValueError(f"Sigmar1 must be greater than 0, sigmar1={sigmar1}")
    if sigmar2 <= 0:
        raise ValueError(f"Sigmar2 must be greater than 0, sigmar2={sigmar2}")
    if sigmas1 <= 0:
        raise ValueError(f"Sigmas1 must be greater than 0, sigmas1={sigmas1}")
    if sigmas2 <= 0:
        raise ValueError(f"Sigmas2 must be greater than 0, sigmas2={sigmas2}")
    prob = 0.25 * (xp.power(2 / xp.pi) * (kappas * xp.exp(-xp.power(xx - mus1, 2) / (2 * sigmas1 ** 2)) / sigmas1 + (1 - kappas) * xp.exp(-xp.power(xx - mus2, 2) / (2 * sigmas2 ** 2)) / sigmas2) * (kappar * (1 - erf((xx - mur1) / (2 ** 0.5 * sigmar1))) + (1 - kappar) * (1 - erf((xx - mur2) / (2 ** 0.5 * sigmar2)))) + xp.power(2 /xp.pi) * (kappar * xp.exp(-xp.power(xx - mur1, 2) / (2 * sigmar1 ** 2)) / sigmar1 + (1 - kappar) * xp.exp(-xp.power(xx - mur2, 2) / (2 * sigmar2 ** 2)) / sigmar2) * (kappas * (1 - erf((xx - mus1) / (2 ** 0.5 *sigmas1))) + (1- kappas) * (1 - erf((xx - mus2) / (2 ** 0.5 * sigmas2)))))
    norm = sum(xx * prob)
    prob /= norm
    prob *= (xx<= high) & (xx >= low)
    return prob


def get_version_information():
    version_file = os.path.join(os.path.dirname(__file__), ".version")
    try:
        with open(version_file, "r") as f:
            return f.readline().rstrip()
    except EnvironmentError:
        print("No version information file '.version' found")
