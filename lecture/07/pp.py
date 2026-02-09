import numba
import numpy as np
from pynucastro.constants import constants
from numba.experimental import jitclass

from pynucastro.rates import (TableIndex, TableInterpolator, TabularRate,
                              TempTableInterpolator, TemperatureTabularRate,
                              Tfactors)
from pynucastro.screening import PlasmaState, ScreenFactors

jp = 0
jd = 1
jhe3 = 2
jhe4 = 3
jli7 = 4
jbe7 = 5
jbe8 = 6
jb8 = 7
nnuc = 8

A = np.zeros((nnuc), dtype=np.int32)

A[jp] = 1
A[jd] = 2
A[jhe3] = 3
A[jhe4] = 4
A[jli7] = 7
A[jbe7] = 7
A[jbe8] = 8
A[jb8] = 8

Z = np.zeros((nnuc), dtype=np.int32)

Z[jp] = 1
Z[jd] = 1
Z[jhe3] = 2
Z[jhe4] = 2
Z[jli7] = 3
Z[jbe7] = 4
Z[jbe8] = 4
Z[jb8] = 5

# masses in ergs
mass = np.zeros((nnuc), dtype=np.float64)

mass[jp] = 0.0015040963047307696
mass[jd] = 0.0030058819195053215
mass[jhe3] = 0.004501176706825056
mass[jhe4] = 0.0059735574859708365
mass[jli7] = 0.010470810414554471
mass[jbe7] = 0.010472191322584432
mass[jbe8] = 0.01194726211305595
mass[jb8] = 0.011976069136782909

names = []
names.append("H1")
names.append("H2")
names.append("He3")
names.append("He4")
names.append("Li7")
names.append("Be7")
names.append("Be8")
names.append("B8")

def to_composition(Y):
    """Convert an array of molar fractions to a Composition object."""
    from pynucastro import Composition, Nucleus
    nuclei = [Nucleus.from_cache(name) for name in names]
    comp = Composition(nuclei)
    for i, nuc in enumerate(nuclei):
        comp.X[nuc] = Y[i] * A[i]
    return comp


def energy_release(dY):
    """return the energy release in erg/g (/s if dY is actually dY/dt)"""
    enuc = 0.0
    for i, y in enumerate(dY):
        enuc += y * mass[i]
    enuc *= -1*constants.N_A
    return enuc

@jitclass([
    ("Be7_to_Li7_reaclib", numba.float64),
    ("B8_to_Be8_reaclib", numba.float64),
    ("B8_to_He4_He4_reaclib", numba.float64),
    ("p_p_to_d_reaclib_bet_pos", numba.float64),
    ("p_p_to_d_reaclib_electron_capture", numba.float64),
    ("p_d_to_He3_reaclib", numba.float64),
    ("d_d_to_He4_reaclib", numba.float64),
    ("p_He3_to_He4_reaclib", numba.float64),
    ("He4_He3_to_Be7_reaclib", numba.float64),
    ("p_Be7_to_B8_reaclib", numba.float64),
    ("d_He3_to_p_He4_reaclib", numba.float64),
    ("p_Li7_to_He4_He4_reaclib", numba.float64),
    ("He3_He3_to_p_p_He4_reaclib", numba.float64),
    ("d_Be7_to_p_He4_He4_reaclib", numba.float64),
    ("He3_Be7_to_p_p_He4_He4_reaclib", numba.float64),
    ("He3_to_p_d_derived", numba.float64),
    ("He4_to_d_d_derived", numba.float64),
    ("Be7_to_He4_He3_derived", numba.float64),
    ("B8_to_p_Be7_derived", numba.float64),
    ("p_He4_to_d_He3_derived", numba.float64),
    ("He4_He4_to_p_Li7_derived", numba.float64),
    ("p_p_He4_to_He3_He3_derived", numba.float64),
    ("p_He4_He4_to_d_Be7_derived", numba.float64),
    ("p_p_He4_He4_to_He3_Be7_derived", numba.float64),
])
class RateEval:
    def __init__(self):
        self.Be7_to_Li7_reaclib = np.nan
        self.B8_to_Be8_reaclib = np.nan
        self.B8_to_He4_He4_reaclib = np.nan
        self.p_p_to_d_reaclib_bet_pos = np.nan
        self.p_p_to_d_reaclib_electron_capture = np.nan
        self.p_d_to_He3_reaclib = np.nan
        self.d_d_to_He4_reaclib = np.nan
        self.p_He3_to_He4_reaclib = np.nan
        self.He4_He3_to_Be7_reaclib = np.nan
        self.p_Be7_to_B8_reaclib = np.nan
        self.d_He3_to_p_He4_reaclib = np.nan
        self.p_Li7_to_He4_He4_reaclib = np.nan
        self.He3_He3_to_p_p_He4_reaclib = np.nan
        self.d_Be7_to_p_He4_He4_reaclib = np.nan
        self.He3_Be7_to_p_p_He4_He4_reaclib = np.nan
        self.He3_to_p_d_derived = np.nan
        self.He4_to_d_d_derived = np.nan
        self.Be7_to_He4_He3_derived = np.nan
        self.B8_to_p_Be7_derived = np.nan
        self.p_He4_to_d_He3_derived = np.nan
        self.He4_He4_to_p_Li7_derived = np.nan
        self.p_p_He4_to_He3_He3_derived = np.nan
        self.p_He4_He4_to_d_Be7_derived = np.nan
        self.p_p_He4_He4_to_He3_Be7_derived = np.nan

@numba.njit()
def ye(Y):
    return np.sum(Z * Y)/np.sum(A * Y)

@numba.njit()
def Be7_to_Li7_reaclib(rate_eval, tf):
    # Be7 --> Li7
    rate = 0.0

    #   ecw
    rate += np.exp(  -23.8328 + 3.02033*tf.T913
                  + -0.0742132*tf.T9 + -0.00792386*tf.T953 + -0.650113*tf.lnT9)

    rate_eval.Be7_to_Li7_reaclib = rate

@numba.njit()
def B8_to_Be8_reaclib(rate_eval, tf):
    # B8 --> Be8
    rate = 0.0

    # wc17w
    rate += np.exp(  -115.234)

    rate_eval.B8_to_Be8_reaclib = rate

@numba.njit()
def B8_to_He4_He4_reaclib(rate_eval, tf):
    # B8 --> He4 + He4
    rate = 0.0

    # wc12w
    rate += np.exp(  -0.105148)

    rate_eval.B8_to_He4_He4_reaclib = rate

@numba.njit()
def p_p_to_d_reaclib_bet_pos(rate_eval, tf):
    # p + p --> d
    rate = 0.0

    # bet+w
    rate += np.exp(  -34.7863 + -3.51193*tf.T913i + 3.10086*tf.T913
                  + -0.198314*tf.T9 + 0.0126251*tf.T953 + -1.02517*tf.lnT9)

    rate_eval.p_p_to_d_reaclib_bet_pos = rate

@numba.njit()
def p_p_to_d_reaclib_electron_capture(rate_eval, tf):
    # p + p --> d
    rate = 0.0

    #   ecw
    rate += np.exp(  -43.6499 + -0.00246064*tf.T9i + -2.7507*tf.T913i + -0.424877*tf.T913
                  + 0.015987*tf.T9 + -0.000690875*tf.T953 + -0.207625*tf.lnT9)

    rate_eval.p_p_to_d_reaclib_electron_capture = rate

@numba.njit()
def p_d_to_He3_reaclib(rate_eval, tf):
    # d + p --> He3
    rate = 0.0

    # de04 
    rate += np.exp(  8.93525 + -3.7208*tf.T913i + 0.198654*tf.T913
                  + 0.333333*tf.lnT9)
    # de04n
    rate += np.exp(  7.52898 + -3.7208*tf.T913i + 0.871782*tf.T913
                  + -0.666667*tf.lnT9)

    rate_eval.p_d_to_He3_reaclib = rate

@numba.njit()
def d_d_to_He4_reaclib(rate_eval, tf):
    # d + d --> He4
    rate = 0.0

    # nacrn
    rate += np.exp(  3.78177 + -4.26166*tf.T913i + -0.119233*tf.T913
                  + 0.778829*tf.T9 + -0.0925203*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.d_d_to_He4_reaclib = rate

@numba.njit()
def p_He3_to_He4_reaclib(rate_eval, tf):
    # He3 + p --> He4
    rate = 0.0

    # bet+w
    rate += np.exp(  -27.7611 + -4.30107e-12*tf.T9i + -6.141*tf.T913i + -1.93473e-09*tf.T913
                  + 2.04145e-10*tf.T9 + -1.80372e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_He3_to_He4_reaclib = rate

@numba.njit()
def He4_He3_to_Be7_reaclib(rate_eval, tf):
    # He3 + He4 --> Be7
    rate = 0.0

    # cd08n
    rate += np.exp(  17.7075 + -12.8271*tf.T913i + -3.8126*tf.T913
                  + 0.0942285*tf.T9 + -0.00301018*tf.T953 + 1.33333*tf.lnT9)
    # cd08n
    rate += np.exp(  15.6099 + -12.8271*tf.T913i + -0.0308225*tf.T913
                  + -0.654685*tf.T9 + 0.0896331*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_He3_to_Be7_reaclib = rate

@numba.njit()
def p_Be7_to_B8_reaclib(rate_eval, tf):
    # Be7 + p --> B8
    rate = 0.0

    # nacrr
    rate += np.exp(  7.73399 + -7.345*tf.T9i
                  + -1.5*tf.lnT9)
    # nacrn
    rate += np.exp(  12.5315 + -10.264*tf.T913i + -0.203472*tf.T913
                  + 0.121083*tf.T9 + -0.00700063*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Be7_to_B8_reaclib = rate

@numba.njit()
def d_He3_to_p_He4_reaclib(rate_eval, tf):
    # He3 + d --> p + He4
    rate = 0.0

    # de04 
    rate += np.exp(  41.2969 + -7.182*tf.T913i + -17.1349*tf.T913
                  + 1.36908*tf.T9 + -0.0814423*tf.T953 + 3.35395*tf.lnT9)
    # de04 
    rate += np.exp(  24.6839 + -7.182*tf.T913i + 0.473288*tf.T913
                  + 1.46847*tf.T9 + -27.9603*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.d_He3_to_p_He4_reaclib = rate

@numba.njit()
def p_Li7_to_He4_He4_reaclib(rate_eval, tf):
    # Li7 + p --> He4 + He4
    rate = 0.0

    # de04r
    rate += np.exp(  21.8999 + -26.1527*tf.T9i
                  + -1.5*tf.lnT9)
    # de04 
    rate += np.exp(  20.4438 + -8.4727*tf.T913i + 0.297934*tf.T913
                  + 0.0582335*tf.T9 + -0.00413383*tf.T953 + -0.666667*tf.lnT9)
    # de04r
    rate += np.exp(  14.2538 + -4.478*tf.T9i
                  + -1.5*tf.lnT9)
    # de04 
    rate += np.exp(  11.9576 + -8.4727*tf.T913i + 0.417943*tf.T913
                  + 5.34565*tf.T9 + -4.8684*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Li7_to_He4_He4_reaclib = rate

@numba.njit()
def He3_He3_to_p_p_He4_reaclib(rate_eval, tf):
    # He3 + He3 --> p + p + He4
    rate = 0.0

    # nacrn
    rate += np.exp(  24.7788 + -12.277*tf.T913i + -0.103699*tf.T913
                  + -0.0649967*tf.T9 + 0.0168191*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He3_He3_to_p_p_He4_reaclib = rate

@numba.njit()
def d_Be7_to_p_He4_He4_reaclib(rate_eval, tf):
    # Be7 + d --> p + He4 + He4
    rate = 0.0

    # cf88n
    rate += np.exp(  27.6987 + -12.428*tf.T913i
                  + -0.666667*tf.lnT9)

    rate_eval.d_Be7_to_p_He4_He4_reaclib = rate

@numba.njit()
def He3_Be7_to_p_p_He4_He4_reaclib(rate_eval, tf):
    # Be7 + He3 --> p + p + He4 + He4
    rate = 0.0

    # mafon
    rate += np.exp(  31.7435 + -5.45213e-12*tf.T9i + -21.793*tf.T913i + -1.98126e-09*tf.T913
                  + 1.84204e-10*tf.T9 + -1.46403e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He3_Be7_to_p_p_He4_He4_reaclib = rate

@numba.njit()
def He3_to_p_d_derived(rate_eval, tf):
    # He3 --> p + d

    rate = 0.0

    # de04 
    rate += np.exp(  32.462003866933586 + -63.74913110454077*tf.T9i + -3.7208*tf.T913i + 0.198654*tf.T913
                  + 1.833333*tf.lnT9)
    # de04n
    rate += np.exp(  31.055733866933583 + -63.74913110454077*tf.T9i + -3.7208*tf.T913i + 0.871782*tf.T913
                  + 0.833333*tf.lnT9)

    rate_eval.He3_to_p_d_derived = rate

    # setting He3 partition function to 1.0 by default, independent of T
    He3_pf = 1.0

    # setting p partition function to 1.0 by default, independent of T
    p_pf = 1.0

    # setting d partition function to 1.0 by default, independent of T
    d_pf = 1.0

    z_r = p_pf*d_pf
    z_p = He3_pf
    rate_eval.He3_to_p_d_derived *= z_r / z_p

@numba.njit()
def He4_to_d_d_derived(rate_eval, tf):
    # He4 --> d + d

    rate = 0.0

    # nacrn
    rate += np.exp(  28.33196127280992 + -276.72748859272605*tf.T9i + -4.26166*tf.T913i + -0.119233*tf.T913
                  + 0.778829*tf.T9 + -0.0925203*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.He4_to_d_d_derived = rate

    # setting He4 partition function to 1.0 by default, independent of T
    He4_pf = 1.0

    # setting d partition function to 1.0 by default, independent of T
    d_pf = 1.0

    z_r = d_pf*d_pf
    z_p = He4_pf
    rate_eval.He4_to_d_d_derived *= z_r / z_p

@numba.njit()
def Be7_to_He4_He3_derived(rate_eval, tf):
    # Be7 --> He4 + He3

    rate = 0.0

    # cd08n
    rate += np.exp(  40.844367255192516 + -18.417933967722725*tf.T9i + -12.8271*tf.T913i + -3.8126*tf.T913
                  + 0.0942285*tf.T9 + -0.00301018*tf.T953 + 2.83333*tf.lnT9)
    # cd08n
    rate += np.exp(  38.746767255192516 + -18.417933967722725*tf.T9i + -12.8271*tf.T913i + -0.0308225*tf.T913
                  + -0.654685*tf.T9 + 0.0896331*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Be7_to_He4_He3_derived = rate

    # setting He3 partition function to 1.0 by default, independent of T
    He3_pf = 1.0

    # setting He4 partition function to 1.0 by default, independent of T
    He4_pf = 1.0

    # setting Be7 partition function to 1.0 by default, independent of T
    Be7_pf = 1.0

    z_r = He4_pf*He3_pf
    z_p = Be7_pf
    rate_eval.Be7_to_He4_He3_derived *= z_r / z_p

@numba.njit()
def B8_to_p_Be7_derived(rate_eval, tf):
    # B8 --> p + Be7

    rate = 0.0

    # nacrr
    rate += np.exp(  31.03415329791634 + -8.927520483432566*tf.T9i)
    # nacrn
    rate += np.exp(  35.83166329791634 + -1.582520483432567*tf.T9i + -10.264*tf.T913i + -0.203472*tf.T913
                  + 0.121083*tf.T9 + -0.00700063*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.B8_to_p_Be7_derived = rate

    # setting p partition function to 1.0 by default, independent of T
    p_pf = 1.0

    # setting Be7 partition function to 1.0 by default, independent of T
    Be7_pf = 1.0

    # setting B8 partition function to 1.0 by default, independent of T
    B8_pf = 1.0

    z_r = p_pf*Be7_pf
    z_p = B8_pf
    rate_eval.B8_to_p_Be7_derived *= z_r / z_p

@numba.njit()
def p_He4_to_d_He3_derived(rate_eval, tf):
    # p + He4 --> d + He3

    rate = 0.0

    # de04 
    rate += np.exp(  43.013484586436284 + -212.97835748819008*tf.T9i + -7.182*tf.T913i + -17.1349*tf.T913
                  + 1.36908*tf.T9 + -0.0814423*tf.T953 + 3.35395*tf.lnT9)
    # de04 
    rate += np.exp(  26.40048458643628 + -212.97835748819008*tf.T9i + -7.182*tf.T913i + 0.473288*tf.T913
                  + 1.46847*tf.T9 + -27.9603*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_He4_to_d_He3_derived = rate

    # setting He3 partition function to 1.0 by default, independent of T
    He3_pf = 1.0

    # setting p partition function to 1.0 by default, independent of T
    p_pf = 1.0

    # setting d partition function to 1.0 by default, independent of T
    d_pf = 1.0

    # setting He4 partition function to 1.0 by default, independent of T
    He4_pf = 1.0

    z_r = d_pf*He3_pf
    z_p = p_pf*He4_pf
    rate_eval.p_He4_to_d_He3_derived *= z_r / z_p

@numba.njit()
def He4_He4_to_p_Li7_derived(rate_eval, tf):
    # He4 + He4 --> p + Li7

    rate = 0.0

    # de04r
    rate += np.exp(  23.454413279927294 + -227.44750659869047*tf.T9i
                  + -1.5*tf.lnT9)
    # de04 
    rate += np.exp(  21.998313279927295 + -201.29480659869046*tf.T9i + -8.4727*tf.T913i + 0.297934*tf.T913
                  + 0.0582335*tf.T9 + -0.00413383*tf.T953 + -0.666667*tf.lnT9)
    # de04r
    rate += np.exp(  15.808313279927294 + -205.77280659869047*tf.T9i
                  + -1.5*tf.lnT9)
    # de04 
    rate += np.exp(  13.512113279927293 + -201.29480659869046*tf.T9i + -8.4727*tf.T913i + 0.417943*tf.T913
                  + 5.34565*tf.T9 + -4.8684*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_He4_to_p_Li7_derived = rate

    # setting Li7 partition function to 1.0 by default, independent of T
    Li7_pf = 1.0

    # setting p partition function to 1.0 by default, independent of T
    p_pf = 1.0

    # setting He4 partition function to 1.0 by default, independent of T
    He4_pf = 1.0

    z_r = p_pf*Li7_pf
    z_p = He4_pf*He4_pf
    rate_eval.He4_He4_to_p_Li7_derived *= z_r / z_p

@numba.njit()
def p_p_He4_to_He3_He3_derived(rate_eval, tf):
    # p + p + He4 --> He3 + He3

    rate = 0.0

    # nacrn
    rate += np.exp(  2.9686307195026984 + -149.2292263836445*tf.T9i + -12.277*tf.T913i + -0.103699*tf.T913
                  + -0.0649967*tf.T9 + 0.0168191*tf.T953 + -2.166667*tf.lnT9)

    rate_eval.p_p_He4_to_He3_He3_derived = rate

    # setting He3 partition function to 1.0 by default, independent of T
    He3_pf = 1.0

    # setting p partition function to 1.0 by default, independent of T
    p_pf = 1.0

    # setting He4 partition function to 1.0 by default, independent of T
    He4_pf = 1.0

    z_r = He3_pf*He3_pf
    z_p = p_pf*p_pf*He4_pf
    rate_eval.p_p_He4_to_He3_He3_derived *= z_r / z_p

@numba.njit()
def p_He4_He4_to_d_Be7_derived(rate_eval, tf):
    # p + He4 + He4 --> d + Be7

    rate = 0.0

    # cf88n
    rate += np.exp(  6.9715645118037095 + -194.56042352046737*tf.T9i + -12.428*tf.T913i
                  + -2.166667*tf.lnT9)

    rate_eval.p_He4_He4_to_d_Be7_derived = rate

    # setting p partition function to 1.0 by default, independent of T
    p_pf = 1.0

    # setting d partition function to 1.0 by default, independent of T
    d_pf = 1.0

    # setting Be7 partition function to 1.0 by default, independent of T
    Be7_pf = 1.0

    # setting He4 partition function to 1.0 by default, independent of T
    He4_pf = 1.0

    z_r = d_pf*Be7_pf
    z_p = p_pf*He4_pf*He4_pf
    rate_eval.p_He4_He4_to_d_Be7_derived *= z_r / z_p

@numba.njit()
def p_p_He4_He4_to_He3_Be7_derived(rate_eval, tf):
    # p + p + He4 + He4 --> He3 + Be7

    rate = 0.0

    # mafon
    rate += np.exp(  -11.817242174569923 + -130.81129241593683*tf.T9i + -21.793*tf.T913i + -1.98126e-09*tf.T913
                  + 1.84204e-10*tf.T9 + -1.46403e-11*tf.T953 + -3.666667*tf.lnT9)

    rate_eval.p_p_He4_He4_to_He3_Be7_derived = rate

    # setting He3 partition function to 1.0 by default, independent of T
    He3_pf = 1.0

    # setting p partition function to 1.0 by default, independent of T
    p_pf = 1.0

    # setting Be7 partition function to 1.0 by default, independent of T
    Be7_pf = 1.0

    # setting He4 partition function to 1.0 by default, independent of T
    He4_pf = 1.0

    z_r = He3_pf*Be7_pf
    z_p = p_pf*p_pf*He4_pf*He4_pf
    rate_eval.p_p_He4_He4_to_He3_Be7_derived *= z_r / z_p

def rhs(t, Y, rho, T, screen_func=None):
    return rhs_eq(t, Y, rho, T, screen_func)

@numba.njit()
def rhs_eq(t, Y, rho, T, screen_func):

    tf = Tfactors(T)
    rate_eval = RateEval()

    # reaclib rates
    Be7_to_Li7_reaclib(rate_eval, tf)
    B8_to_Be8_reaclib(rate_eval, tf)
    B8_to_He4_He4_reaclib(rate_eval, tf)
    p_p_to_d_reaclib_bet_pos(rate_eval, tf)
    p_p_to_d_reaclib_electron_capture(rate_eval, tf)
    p_d_to_He3_reaclib(rate_eval, tf)
    d_d_to_He4_reaclib(rate_eval, tf)
    p_He3_to_He4_reaclib(rate_eval, tf)
    He4_He3_to_Be7_reaclib(rate_eval, tf)
    p_Be7_to_B8_reaclib(rate_eval, tf)
    d_He3_to_p_He4_reaclib(rate_eval, tf)
    p_Li7_to_He4_He4_reaclib(rate_eval, tf)
    He3_He3_to_p_p_He4_reaclib(rate_eval, tf)
    d_Be7_to_p_He4_He4_reaclib(rate_eval, tf)
    He3_Be7_to_p_p_He4_He4_reaclib(rate_eval, tf)

    # derived rates
    He3_to_p_d_derived(rate_eval, tf)
    He4_to_d_d_derived(rate_eval, tf)
    Be7_to_He4_He3_derived(rate_eval, tf)
    B8_to_p_Be7_derived(rate_eval, tf)
    p_He4_to_d_He3_derived(rate_eval, tf)
    He4_He4_to_p_Li7_derived(rate_eval, tf)
    p_p_He4_to_He3_He3_derived(rate_eval, tf)
    p_He4_He4_to_d_Be7_derived(rate_eval, tf)
    p_p_He4_He4_to_He3_Be7_derived(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

        scn_fac = ScreenFactors(1, 1, 1, 1)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_p_to_d_reaclib_bet_pos *= scor
        rate_eval.p_p_to_d_reaclib_electron_capture *= scor
        rate_eval.p_p_He4_He4_to_He3_Be7_derived *= scor

        scn_fac = ScreenFactors(1, 1, 1, 2)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_d_to_He3_reaclib *= scor

        scn_fac = ScreenFactors(1, 2, 1, 2)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_d_to_He4_reaclib *= scor

        scn_fac = ScreenFactors(1, 1, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_He3_to_He4_reaclib *= scor

        scn_fac = ScreenFactors(2, 4, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_He3_to_Be7_reaclib *= scor

        scn_fac = ScreenFactors(1, 1, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Be7_to_B8_reaclib *= scor

        scn_fac = ScreenFactors(1, 2, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_He3_to_p_He4_reaclib *= scor

        scn_fac = ScreenFactors(1, 1, 3, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Li7_to_He4_He4_reaclib *= scor

        scn_fac = ScreenFactors(2, 3, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He3_He3_to_p_p_He4_reaclib *= scor

        scn_fac = ScreenFactors(1, 2, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_Be7_to_p_He4_He4_reaclib *= scor

        scn_fac = ScreenFactors(2, 3, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He3_Be7_to_p_p_He4_He4_reaclib *= scor

        scn_fac = ScreenFactors(1, 1, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_He4_to_d_He3_derived *= scor

        scn_fac = ScreenFactors(2, 4, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_He4_to_p_Li7_derived *= scor

        scn_fac = ScreenFactors(1, 1, 1, 1)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_p_He4_to_He3_He3_derived *= scor

        scn_fac = ScreenFactors(1, 1, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_He4_He4_to_d_Be7_derived *= scor

    dYdt = np.zeros((nnuc), dtype=np.float64)

    dYdt[jp] = (
          + -2*5.00000000000000e-01*rho*Y[jp]**2*rate_eval.p_p_to_d_reaclib_bet_pos  +
          + -2*5.00000000000000e-01*rho**2*ye(Y)*Y[jp]**2*rate_eval.p_p_to_d_reaclib_electron_capture  +
          ( -rho*Y[jp]*Y[jd]*rate_eval.p_d_to_He3_reaclib +Y[jhe3]*rate_eval.He3_to_p_d_derived ) +
          -rho*Y[jp]*Y[jhe3]*rate_eval.p_He3_to_He4_reaclib  +
          ( -rho*Y[jp]*Y[jbe7]*rate_eval.p_Be7_to_B8_reaclib +Y[jb8]*rate_eval.B8_to_p_Be7_derived ) +
          ( +rho*Y[jd]*Y[jhe3]*rate_eval.d_He3_to_p_He4_reaclib -rho*Y[jp]*Y[jhe4]*rate_eval.p_He4_to_d_He3_derived ) +
          ( -rho*Y[jp]*Y[jli7]*rate_eval.p_Li7_to_He4_He4_reaclib +5.00000000000000e-01*rho*Y[jhe4]**2*rate_eval.He4_He4_to_p_Li7_derived ) +
          ( + 2*5.00000000000000e-01*rho*Y[jhe3]**2*rate_eval.He3_He3_to_p_p_He4_reaclib + -2*5.00000000000000e-01*rho**2*Y[jp]**2*Y[jhe4]*rate_eval.p_p_He4_to_He3_He3_derived ) +
          ( +rho*Y[jd]*Y[jbe7]*rate_eval.d_Be7_to_p_He4_He4_reaclib -5.00000000000000e-01*rho**2*Y[jp]*Y[jhe4]**2*rate_eval.p_He4_He4_to_d_Be7_derived ) +
          ( + 2*rho*Y[jhe3]*Y[jbe7]*rate_eval.He3_Be7_to_p_p_He4_He4_reaclib + -2*2.50000000000000e-01*rho**3*Y[jp]**2*Y[jhe4]**2*rate_eval.p_p_He4_He4_to_He3_Be7_derived )
       )

    dYdt[jd] = (
          +5.00000000000000e-01*rho*Y[jp]**2*rate_eval.p_p_to_d_reaclib_bet_pos  +
          +5.00000000000000e-01*rho**2*ye(Y)*Y[jp]**2*rate_eval.p_p_to_d_reaclib_electron_capture  +
          ( -rho*Y[jp]*Y[jd]*rate_eval.p_d_to_He3_reaclib +Y[jhe3]*rate_eval.He3_to_p_d_derived ) +
          ( + -2*5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d_to_He4_reaclib + 2*Y[jhe4]*rate_eval.He4_to_d_d_derived ) +
          ( -rho*Y[jd]*Y[jhe3]*rate_eval.d_He3_to_p_He4_reaclib +rho*Y[jp]*Y[jhe4]*rate_eval.p_He4_to_d_He3_derived ) +
          ( -rho*Y[jd]*Y[jbe7]*rate_eval.d_Be7_to_p_He4_He4_reaclib +5.00000000000000e-01*rho**2*Y[jp]*Y[jhe4]**2*rate_eval.p_He4_He4_to_d_Be7_derived )
       )

    dYdt[jhe3] = (
          ( +rho*Y[jp]*Y[jd]*rate_eval.p_d_to_He3_reaclib -Y[jhe3]*rate_eval.He3_to_p_d_derived ) +
          -rho*Y[jp]*Y[jhe3]*rate_eval.p_He3_to_He4_reaclib  +
          ( -rho*Y[jhe3]*Y[jhe4]*rate_eval.He4_He3_to_Be7_reaclib +Y[jbe7]*rate_eval.Be7_to_He4_He3_derived ) +
          ( -rho*Y[jd]*Y[jhe3]*rate_eval.d_He3_to_p_He4_reaclib +rho*Y[jp]*Y[jhe4]*rate_eval.p_He4_to_d_He3_derived ) +
          ( + -2*5.00000000000000e-01*rho*Y[jhe3]**2*rate_eval.He3_He3_to_p_p_He4_reaclib + 2*5.00000000000000e-01*rho**2*Y[jp]**2*Y[jhe4]*rate_eval.p_p_He4_to_He3_He3_derived ) +
          ( -rho*Y[jhe3]*Y[jbe7]*rate_eval.He3_Be7_to_p_p_He4_He4_reaclib +2.50000000000000e-01*rho**3*Y[jp]**2*Y[jhe4]**2*rate_eval.p_p_He4_He4_to_He3_Be7_derived )
       )

    dYdt[jhe4] = (
          + 2*Y[jb8]*rate_eval.B8_to_He4_He4_reaclib  +
          ( +5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d_to_He4_reaclib -Y[jhe4]*rate_eval.He4_to_d_d_derived ) +
          +rho*Y[jp]*Y[jhe3]*rate_eval.p_He3_to_He4_reaclib  +
          ( -rho*Y[jhe3]*Y[jhe4]*rate_eval.He4_He3_to_Be7_reaclib +Y[jbe7]*rate_eval.Be7_to_He4_He3_derived ) +
          ( +rho*Y[jd]*Y[jhe3]*rate_eval.d_He3_to_p_He4_reaclib -rho*Y[jp]*Y[jhe4]*rate_eval.p_He4_to_d_He3_derived ) +
          ( + 2*rho*Y[jp]*Y[jli7]*rate_eval.p_Li7_to_He4_He4_reaclib + -2*5.00000000000000e-01*rho*Y[jhe4]**2*rate_eval.He4_He4_to_p_Li7_derived ) +
          ( +5.00000000000000e-01*rho*Y[jhe3]**2*rate_eval.He3_He3_to_p_p_He4_reaclib -5.00000000000000e-01*rho**2*Y[jp]**2*Y[jhe4]*rate_eval.p_p_He4_to_He3_He3_derived ) +
          ( + 2*rho*Y[jd]*Y[jbe7]*rate_eval.d_Be7_to_p_He4_He4_reaclib + -2*5.00000000000000e-01*rho**2*Y[jp]*Y[jhe4]**2*rate_eval.p_He4_He4_to_d_Be7_derived ) +
          ( + 2*rho*Y[jhe3]*Y[jbe7]*rate_eval.He3_Be7_to_p_p_He4_He4_reaclib + -2*2.50000000000000e-01*rho**3*Y[jp]**2*Y[jhe4]**2*rate_eval.p_p_He4_He4_to_He3_Be7_derived )
       )

    dYdt[jli7] = (
          +rho*ye(Y)*Y[jbe7]*rate_eval.Be7_to_Li7_reaclib  +
          ( -rho*Y[jp]*Y[jli7]*rate_eval.p_Li7_to_He4_He4_reaclib +5.00000000000000e-01*rho*Y[jhe4]**2*rate_eval.He4_He4_to_p_Li7_derived )
       )

    dYdt[jbe7] = (
          -rho*ye(Y)*Y[jbe7]*rate_eval.Be7_to_Li7_reaclib  +
          ( +rho*Y[jhe3]*Y[jhe4]*rate_eval.He4_He3_to_Be7_reaclib -Y[jbe7]*rate_eval.Be7_to_He4_He3_derived ) +
          ( -rho*Y[jp]*Y[jbe7]*rate_eval.p_Be7_to_B8_reaclib +Y[jb8]*rate_eval.B8_to_p_Be7_derived ) +
          ( -rho*Y[jd]*Y[jbe7]*rate_eval.d_Be7_to_p_He4_He4_reaclib +5.00000000000000e-01*rho**2*Y[jp]*Y[jhe4]**2*rate_eval.p_He4_He4_to_d_Be7_derived ) +
          ( -rho*Y[jhe3]*Y[jbe7]*rate_eval.He3_Be7_to_p_p_He4_He4_reaclib +2.50000000000000e-01*rho**3*Y[jp]**2*Y[jhe4]**2*rate_eval.p_p_He4_He4_to_He3_Be7_derived )
       )

    dYdt[jbe8] = (
          +Y[jb8]*rate_eval.B8_to_Be8_reaclib
       )

    dYdt[jb8] = (
          -Y[jb8]*rate_eval.B8_to_Be8_reaclib  +
          -Y[jb8]*rate_eval.B8_to_He4_He4_reaclib  +
          ( +rho*Y[jp]*Y[jbe7]*rate_eval.p_Be7_to_B8_reaclib -Y[jb8]*rate_eval.B8_to_p_Be7_derived )
       )

    return dYdt

def jacobian(t, Y, rho, T, screen_func=None):
    return jacobian_eq(t, Y, rho, T, screen_func)

@numba.njit()
def jacobian_eq(t, Y, rho, T, screen_func):

    tf = Tfactors(T)
    rate_eval = RateEval()

    # reaclib rates
    Be7_to_Li7_reaclib(rate_eval, tf)
    B8_to_Be8_reaclib(rate_eval, tf)
    B8_to_He4_He4_reaclib(rate_eval, tf)
    p_p_to_d_reaclib_bet_pos(rate_eval, tf)
    p_p_to_d_reaclib_electron_capture(rate_eval, tf)
    p_d_to_He3_reaclib(rate_eval, tf)
    d_d_to_He4_reaclib(rate_eval, tf)
    p_He3_to_He4_reaclib(rate_eval, tf)
    He4_He3_to_Be7_reaclib(rate_eval, tf)
    p_Be7_to_B8_reaclib(rate_eval, tf)
    d_He3_to_p_He4_reaclib(rate_eval, tf)
    p_Li7_to_He4_He4_reaclib(rate_eval, tf)
    He3_He3_to_p_p_He4_reaclib(rate_eval, tf)
    d_Be7_to_p_He4_He4_reaclib(rate_eval, tf)
    He3_Be7_to_p_p_He4_He4_reaclib(rate_eval, tf)

    # derived rates
    He3_to_p_d_derived(rate_eval, tf)
    He4_to_d_d_derived(rate_eval, tf)
    Be7_to_He4_He3_derived(rate_eval, tf)
    B8_to_p_Be7_derived(rate_eval, tf)
    p_He4_to_d_He3_derived(rate_eval, tf)
    He4_He4_to_p_Li7_derived(rate_eval, tf)
    p_p_He4_to_He3_He3_derived(rate_eval, tf)
    p_He4_He4_to_d_Be7_derived(rate_eval, tf)
    p_p_He4_He4_to_He3_Be7_derived(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

        scn_fac = ScreenFactors(1, 1, 1, 1)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_p_to_d_reaclib_bet_pos *= scor
        rate_eval.p_p_to_d_reaclib_electron_capture *= scor
        rate_eval.p_p_He4_He4_to_He3_Be7_derived *= scor

        scn_fac = ScreenFactors(1, 1, 1, 2)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_d_to_He3_reaclib *= scor

        scn_fac = ScreenFactors(1, 2, 1, 2)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_d_to_He4_reaclib *= scor

        scn_fac = ScreenFactors(1, 1, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_He3_to_He4_reaclib *= scor

        scn_fac = ScreenFactors(2, 4, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_He3_to_Be7_reaclib *= scor

        scn_fac = ScreenFactors(1, 1, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Be7_to_B8_reaclib *= scor

        scn_fac = ScreenFactors(1, 2, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_He3_to_p_He4_reaclib *= scor

        scn_fac = ScreenFactors(1, 1, 3, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Li7_to_He4_He4_reaclib *= scor

        scn_fac = ScreenFactors(2, 3, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He3_He3_to_p_p_He4_reaclib *= scor

        scn_fac = ScreenFactors(1, 2, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_Be7_to_p_He4_He4_reaclib *= scor

        scn_fac = ScreenFactors(2, 3, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He3_Be7_to_p_p_He4_He4_reaclib *= scor

        scn_fac = ScreenFactors(1, 1, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_He4_to_d_He3_derived *= scor

        scn_fac = ScreenFactors(2, 4, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_He4_to_p_Li7_derived *= scor

        scn_fac = ScreenFactors(1, 1, 1, 1)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_p_He4_to_He3_He3_derived *= scor

        scn_fac = ScreenFactors(1, 1, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_He4_He4_to_d_Be7_derived *= scor

    jac = np.zeros((nnuc, nnuc), dtype=np.float64)

    jac[jp, jp] = (
       -2*5.00000000000000e-01*rho*2*Y[jp]*rate_eval.p_p_to_d_reaclib_bet_pos
       -2*5.00000000000000e-01*rho**2*ye(Y)*2*Y[jp]*rate_eval.p_p_to_d_reaclib_electron_capture
       -rho*Y[jd]*rate_eval.p_d_to_He3_reaclib
       -rho*Y[jhe3]*rate_eval.p_He3_to_He4_reaclib
       -rho*Y[jbe7]*rate_eval.p_Be7_to_B8_reaclib
       -rho*Y[jli7]*rate_eval.p_Li7_to_He4_He4_reaclib
       -rho*Y[jhe4]*rate_eval.p_He4_to_d_He3_derived
       -2*5.00000000000000e-01*rho**2*2*Y[jp]*Y[jhe4]*rate_eval.p_p_He4_to_He3_He3_derived
       -5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.p_He4_He4_to_d_Be7_derived
       -2*2.50000000000000e-01*rho**3*2*Y[jp]*Y[jhe4]**2*rate_eval.p_p_He4_He4_to_He3_Be7_derived
       )

    jac[jp, jd] = (
       -rho*Y[jp]*rate_eval.p_d_to_He3_reaclib
       +rho*Y[jhe3]*rate_eval.d_He3_to_p_He4_reaclib
       +rho*Y[jbe7]*rate_eval.d_Be7_to_p_He4_He4_reaclib
       )

    jac[jp, jhe3] = (
       -rho*Y[jp]*rate_eval.p_He3_to_He4_reaclib
       +rho*Y[jd]*rate_eval.d_He3_to_p_He4_reaclib
       +2*5.00000000000000e-01*rho*2*Y[jhe3]*rate_eval.He3_He3_to_p_p_He4_reaclib
       +2*rho*Y[jbe7]*rate_eval.He3_Be7_to_p_p_He4_He4_reaclib
       +rate_eval.He3_to_p_d_derived
       )

    jac[jp, jhe4] = (
       -rho*Y[jp]*rate_eval.p_He4_to_d_He3_derived
       -2*5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.p_p_He4_to_He3_He3_derived
       -5.00000000000000e-01*rho**2*Y[jp]*2*Y[jhe4]*rate_eval.p_He4_He4_to_d_Be7_derived
       -2*2.50000000000000e-01*rho**3*Y[jp]**2*2*Y[jhe4]*rate_eval.p_p_He4_He4_to_He3_Be7_derived
       +5.00000000000000e-01*rho*2*Y[jhe4]*rate_eval.He4_He4_to_p_Li7_derived
       )

    jac[jp, jli7] = (
       -rho*Y[jp]*rate_eval.p_Li7_to_He4_He4_reaclib
       )

    jac[jp, jbe7] = (
       -rho*Y[jp]*rate_eval.p_Be7_to_B8_reaclib
       +rho*Y[jd]*rate_eval.d_Be7_to_p_He4_He4_reaclib
       +2*rho*Y[jhe3]*rate_eval.He3_Be7_to_p_p_He4_He4_reaclib
       )

    jac[jp, jb8] = (
       +rate_eval.B8_to_p_Be7_derived
       )

    jac[jd, jp] = (
       -rho*Y[jd]*rate_eval.p_d_to_He3_reaclib
       +5.00000000000000e-01*rho*2*Y[jp]*rate_eval.p_p_to_d_reaclib_bet_pos
       +5.00000000000000e-01*rho**2*ye(Y)*2*Y[jp]*rate_eval.p_p_to_d_reaclib_electron_capture
       +rho*Y[jhe4]*rate_eval.p_He4_to_d_He3_derived
       +5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.p_He4_He4_to_d_Be7_derived
       )

    jac[jd, jd] = (
       -rho*Y[jp]*rate_eval.p_d_to_He3_reaclib
       -2*5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d_to_He4_reaclib
       -rho*Y[jhe3]*rate_eval.d_He3_to_p_He4_reaclib
       -rho*Y[jbe7]*rate_eval.d_Be7_to_p_He4_He4_reaclib
       )

    jac[jd, jhe3] = (
       -rho*Y[jd]*rate_eval.d_He3_to_p_He4_reaclib
       +rate_eval.He3_to_p_d_derived
       )

    jac[jd, jhe4] = (
       +2*rate_eval.He4_to_d_d_derived
       +rho*Y[jp]*rate_eval.p_He4_to_d_He3_derived
       +5.00000000000000e-01*rho**2*Y[jp]*2*Y[jhe4]*rate_eval.p_He4_He4_to_d_Be7_derived
       )

    jac[jd, jbe7] = (
       -rho*Y[jd]*rate_eval.d_Be7_to_p_He4_He4_reaclib
       )

    jac[jhe3, jp] = (
       -rho*Y[jhe3]*rate_eval.p_He3_to_He4_reaclib
       +rho*Y[jd]*rate_eval.p_d_to_He3_reaclib
       +rho*Y[jhe4]*rate_eval.p_He4_to_d_He3_derived
       +2*5.00000000000000e-01*rho**2*2*Y[jp]*Y[jhe4]*rate_eval.p_p_He4_to_He3_He3_derived
       +2.50000000000000e-01*rho**3*2*Y[jp]*Y[jhe4]**2*rate_eval.p_p_He4_He4_to_He3_Be7_derived
       )

    jac[jhe3, jd] = (
       -rho*Y[jhe3]*rate_eval.d_He3_to_p_He4_reaclib
       +rho*Y[jp]*rate_eval.p_d_to_He3_reaclib
       )

    jac[jhe3, jhe3] = (
       -rho*Y[jp]*rate_eval.p_He3_to_He4_reaclib
       -rho*Y[jhe4]*rate_eval.He4_He3_to_Be7_reaclib
       -rho*Y[jd]*rate_eval.d_He3_to_p_He4_reaclib
       -2*5.00000000000000e-01*rho*2*Y[jhe3]*rate_eval.He3_He3_to_p_p_He4_reaclib
       -rho*Y[jbe7]*rate_eval.He3_Be7_to_p_p_He4_He4_reaclib
       -rate_eval.He3_to_p_d_derived
       )

    jac[jhe3, jhe4] = (
       -rho*Y[jhe3]*rate_eval.He4_He3_to_Be7_reaclib
       +rho*Y[jp]*rate_eval.p_He4_to_d_He3_derived
       +2*5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.p_p_He4_to_He3_He3_derived
       +2.50000000000000e-01*rho**3*Y[jp]**2*2*Y[jhe4]*rate_eval.p_p_He4_He4_to_He3_Be7_derived
       )

    jac[jhe3, jbe7] = (
       -rho*Y[jhe3]*rate_eval.He3_Be7_to_p_p_He4_He4_reaclib
       +rate_eval.Be7_to_He4_He3_derived
       )

    jac[jhe4, jp] = (
       -rho*Y[jhe4]*rate_eval.p_He4_to_d_He3_derived
       -5.00000000000000e-01*rho**2*2*Y[jp]*Y[jhe4]*rate_eval.p_p_He4_to_He3_He3_derived
       -2*5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.p_He4_He4_to_d_Be7_derived
       -2*2.50000000000000e-01*rho**3*2*Y[jp]*Y[jhe4]**2*rate_eval.p_p_He4_He4_to_He3_Be7_derived
       +rho*Y[jhe3]*rate_eval.p_He3_to_He4_reaclib
       +2*rho*Y[jli7]*rate_eval.p_Li7_to_He4_He4_reaclib
       )

    jac[jhe4, jd] = (
       +5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d_to_He4_reaclib
       +rho*Y[jhe3]*rate_eval.d_He3_to_p_He4_reaclib
       +2*rho*Y[jbe7]*rate_eval.d_Be7_to_p_He4_He4_reaclib
       )

    jac[jhe4, jhe3] = (
       -rho*Y[jhe4]*rate_eval.He4_He3_to_Be7_reaclib
       +rho*Y[jp]*rate_eval.p_He3_to_He4_reaclib
       +rho*Y[jd]*rate_eval.d_He3_to_p_He4_reaclib
       +5.00000000000000e-01*rho*2*Y[jhe3]*rate_eval.He3_He3_to_p_p_He4_reaclib
       +2*rho*Y[jbe7]*rate_eval.He3_Be7_to_p_p_He4_He4_reaclib
       )

    jac[jhe4, jhe4] = (
       -rho*Y[jhe3]*rate_eval.He4_He3_to_Be7_reaclib
       -rate_eval.He4_to_d_d_derived
       -rho*Y[jp]*rate_eval.p_He4_to_d_He3_derived
       -2*5.00000000000000e-01*rho*2*Y[jhe4]*rate_eval.He4_He4_to_p_Li7_derived
       -5.00000000000000e-01*rho**2*Y[jp]**2*rate_eval.p_p_He4_to_He3_He3_derived
       -2*5.00000000000000e-01*rho**2*Y[jp]*2*Y[jhe4]*rate_eval.p_He4_He4_to_d_Be7_derived
       -2*2.50000000000000e-01*rho**3*Y[jp]**2*2*Y[jhe4]*rate_eval.p_p_He4_He4_to_He3_Be7_derived
       )

    jac[jhe4, jli7] = (
       +2*rho*Y[jp]*rate_eval.p_Li7_to_He4_He4_reaclib
       )

    jac[jhe4, jbe7] = (
       +2*rho*Y[jd]*rate_eval.d_Be7_to_p_He4_He4_reaclib
       +2*rho*Y[jhe3]*rate_eval.He3_Be7_to_p_p_He4_He4_reaclib
       +rate_eval.Be7_to_He4_He3_derived
       )

    jac[jhe4, jb8] = (
       +2*rate_eval.B8_to_He4_He4_reaclib
       )

    jac[jli7, jp] = (
       -rho*Y[jli7]*rate_eval.p_Li7_to_He4_He4_reaclib
       )

    jac[jli7, jhe4] = (
       +5.00000000000000e-01*rho*2*Y[jhe4]*rate_eval.He4_He4_to_p_Li7_derived
       )

    jac[jli7, jli7] = (
       -rho*Y[jp]*rate_eval.p_Li7_to_He4_He4_reaclib
       )

    jac[jli7, jbe7] = (
       +rho*ye(Y)*rate_eval.Be7_to_Li7_reaclib
       )

    jac[jbe7, jp] = (
       -rho*Y[jbe7]*rate_eval.p_Be7_to_B8_reaclib
       +5.00000000000000e-01*rho**2*Y[jhe4]**2*rate_eval.p_He4_He4_to_d_Be7_derived
       +2.50000000000000e-01*rho**3*2*Y[jp]*Y[jhe4]**2*rate_eval.p_p_He4_He4_to_He3_Be7_derived
       )

    jac[jbe7, jd] = (
       -rho*Y[jbe7]*rate_eval.d_Be7_to_p_He4_He4_reaclib
       )

    jac[jbe7, jhe3] = (
       -rho*Y[jbe7]*rate_eval.He3_Be7_to_p_p_He4_He4_reaclib
       +rho*Y[jhe4]*rate_eval.He4_He3_to_Be7_reaclib
       )

    jac[jbe7, jhe4] = (
       +rho*Y[jhe3]*rate_eval.He4_He3_to_Be7_reaclib
       +5.00000000000000e-01*rho**2*Y[jp]*2*Y[jhe4]*rate_eval.p_He4_He4_to_d_Be7_derived
       +2.50000000000000e-01*rho**3*Y[jp]**2*2*Y[jhe4]*rate_eval.p_p_He4_He4_to_He3_Be7_derived
       )

    jac[jbe7, jbe7] = (
       -rho*ye(Y)*rate_eval.Be7_to_Li7_reaclib
       -rho*Y[jp]*rate_eval.p_Be7_to_B8_reaclib
       -rho*Y[jd]*rate_eval.d_Be7_to_p_He4_He4_reaclib
       -rho*Y[jhe3]*rate_eval.He3_Be7_to_p_p_He4_He4_reaclib
       -rate_eval.Be7_to_He4_He3_derived
       )

    jac[jbe7, jb8] = (
       +rate_eval.B8_to_p_Be7_derived
       )

    jac[jbe8, jb8] = (
       +rate_eval.B8_to_Be8_reaclib
       )

    jac[jb8, jp] = (
       +rho*Y[jbe7]*rate_eval.p_Be7_to_B8_reaclib
       )

    jac[jb8, jbe7] = (
       +rho*Y[jp]*rate_eval.p_Be7_to_B8_reaclib
       )

    jac[jb8, jb8] = (
       -rate_eval.B8_to_Be8_reaclib
       -rate_eval.B8_to_He4_He4_reaclib
       -rate_eval.B8_to_p_Be7_derived
       )

    return jac
