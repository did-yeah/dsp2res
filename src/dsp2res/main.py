"""
Programmes pour transformer une DSP en réponse fréquentiel
"""

import math
from collections.abc import Sequence
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np

type V = np.ndarray[tuple[int], np.dtype[np.float64]]


def trapezoid(y: V, x: V | None = None, dx=1.0, axis=-1):
    """
    émule la fonction trpezoid de numpy
    copié de:
        https://github.com/numpy/numpy/blob/main/numpy/lib/_function_base_impl.py#L4949-L5081
    """
    y = np.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = np.asanyarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = np.diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    try:
        ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
    except ValueError:
        # Operations didn't work, cast to ndarray
        d = np.asarray(d)
        y = np.asarray(y)
        ret = np.add.reduce(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
    return ret


def Haf_norm(ksi: float, f0: float, freqs: V) -> V:
    """
    renvoie un array du module au carré de la réponse harmonique d'un oscillateur
    de fréquence propre f0 avec un amortissement adimensionnel ksi aux frequences
    freqs
    >>> res = Haf_norm(0.1, 10, np.array([1., 5., 9., 10., 11., 20., 30.]))
    >>> np.allclose(res, np.array([1., 1.7, 15., 26., 11.3, 0.12, 0.021]), rtol=0.1, atol=0.01)
    True
    """
    r_2 = (freqs / f0) ** 2
    d_ksi_r_2 = r_2 * (2 * ksi) ** 2
    return (1.0 + d_ksi_r_2) / ((1.0 - r_2) ** 2 + d_ksi_r_2)


def res_dsp(ksi: float, freqs: V, dsp_freqs: V, dsp_vals: V) -> V:
    """
    renvoie la moyenne de la réponse en fonction de l'oscillateur harmonique soumis à
    une DSP à chaque fréquence propre
    >>> res =res_dsp(0.1, np.array([1., 5., 9., 10., 11., 20., 30.]), np.array([1., 10., 20.]), np.array([0.1, 1., 0.1]))
    >>> np.allclose(res, np.array([2.28, 7.29, 9.42, 10.73, 12.45, 6.45, 3.91]), rtol=0.01, atol=0.01)
    True
    """
    dsp_val2 = np.interp(freqs, dsp_freqs, dsp_vals, left=0, right=0)
    return np.sqrt(
        np.transpose(
            trapezoid(
                np.array([dsp_val2 * Haf_norm(ksi, f, freqs) for f in freqs]),
                x=freqs,
                axis=1,
            ),
        ),
    )


def res_dsp_log_log(ksi: float, freqs: V, dsp_freqs: V, dsp_vals: V) -> V:
    """
    renvoie la moyenne de la réponse en fonction de l'oscillateur harmonique soumis à
    une DSP à chaque fréquence propre
    applique une interpolation log log
    >>> res =res_dsp_log_log(0.1, np.array([1., 5., 9., 10., 11., 20., 30.]), np.array([1., 10., 20.]), np.array([0.1, 1., 0.1]))
    >>> np.allclose(res, np.array([2.28, 7.28, 9.25, 10.25, 11.47, 6.31, 3.75]), rtol=0.01, atol=0.01)
    True
    """
    dsp_val2 = np.power(
        10.0,
        np.interp(
            np.log10(freqs),
            np.log10(dsp_freqs),
            np.log10(dsp_vals),
            left=-10,
            right=-10,
        ),
    )
    return np.sqrt(
        np.transpose(
            trapezoid(
                np.array([dsp_val2 * Haf_norm(ksi, f, freqs) for f in freqs]),
                x=freqs,
                axis=1,
            ),
        ),
    )


def integral_log_log(f1: float, f2: float, A: float, B: float, ordre: int) -> float:
    """
    calcul de l'intégrale définie dans un diagramme log log
    entre les fréquences f1 et f2
    de niveau A et B
    ordre est le niveau du moment
    """
    ex = (math.log10(B) - math.log10(B)) / (math.log10(f2) - math.log10(f1))
    return (A / ((ex + 1 + ordre) * f1**ex)) * (f2 ** (ordre + 1) - f1 ** (ordre + 1))


def moment_dsp(ordre: int, dsp_freqs: V, dsp_vals: V) -> float:
    """
    calcul d'un moment à l'ordre ordre de la dsp
    """
    res = 0.0
    for (f1, f2), (A, B) in zip(
        pairwise(dsp_freqs),
        pairwise(dsp_vals),
    ):
        BmA = (B - A) / (f2 - f1)
        res += (f2 ** (ordre + 1) - f1 ** (ordre + 1)) * (
            A / (ordre + 1) - (BmA) * f1 / (ordre + 1)
        ) + BmA * (f2 ** (ordre + 2) - f1 ** (ordre + 2)) / (ordre + 2)
    return res


def moment_dsp_log_log(ordre: int, dsp_freqs: V, dsp_vals: V) -> float:
    """
    calcul d'un moment à l'ordre ordre de la dsp
    """
    res = 0.0
    for (f1, f2), (A, B) in zip(
        pairwise(dsp_freqs),
        pairwise(dsp_vals),
    ):
        res += integral_log_log(f1, f2, A, B, ordre)
    return res


def n0_plus(dsp_freqs: V, dsp_vals: V) -> float:
    """
    calcul la fréquence moyenne
    """
    return math.sqrt(
        moment_dsp(2, dsp_freqs, dsp_vals) / moment_dsp(0, dsp_freqs, dsp_vals),
    )


def n0_plus_log_log(dsp_freqs: V, dsp_vals: V) -> float:
    """
    calcul la fréquence moyenne
    """
    return math.sqrt(
        moment_dsp_log_log(2, dsp_freqs, dsp_vals)
        / moment_dsp_log_log(0, dsp_freqs, dsp_vals),
    )


def plt_res_dsp(ksi: Sequence[float], freqs: V, dsp_freqs: V, dsp_vals: V) -> None:
    """
    trace pour plusieurs amortissements les réponse spectrales associée à un DSP
    définie en log-log (décimal)
    l'abscisse des fréquence est données par freqs
    """
    # recherche des couleurs
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    # Création de la figure
    _fig, ax = plt.subplots(figsize=(10, 8))

    # tracé
    for ks, color in zip(ksi, colors):
        ax.plot(
            freqs,
            res_dsp_log_log(ks, freqs, dsp_freqs, dsp_vals),
            color=color,
            label=f"amo = {ks}",
        )
        ax.plot(
            freqs,
            res_dsp(ks, freqs, dsp_freqs, dsp_vals),
            color=color,
            linestyle="dotted",
            label=f"amo = {ks}",
        )

    # mise en forme
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel(r"accel rms")
    ax.set_title("réponse spectrale aux amortissements")
    ax.grid(True)
    ax.legend()

    # affichage
    plt.tight_layout()
    plt.show()


def main() -> None:
    f_dsp = np.array([15.0, 30.0, 100.0, 500.0, 800.0])
    A_dsp = np.array([1e-3, 1.0e-2, 3.0e-2, 3.0e-2, 1.0e-3])
    n0p = n0_plus_log_log(f_dsp, A_dsp)
    n0p_lin = n0_plus(f_dsp, A_dsp)
    moy = math.sqrt(moment_dsp_log_log(0, f_dsp, A_dsp))
    moy_lin = math.sqrt(moment_dsp(0, f_dsp, A_dsp))
    print(f"moyenne = {moy_lin}")
    print(f"n0+ = {n0p_lin}")
    print(f"moyenne log-log= {moy}")
    print(f"n0+ log-log= {n0p}")
    plt_res_dsp(
        [0.02, 0.05, 0.2],
        np.linspace(15.0, 3500.0, 3000),
        f_dsp,
        A_dsp,
    )
