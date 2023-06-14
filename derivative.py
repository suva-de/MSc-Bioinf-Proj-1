# Derivatives of the reaction terms (used in Jacobian function jac_2Node below)
from scipy import *
from sympy import diff, sin, exp 
from sympy import * 



def derivate (f,wrt):

    df = diff(f, wrt)

    return df

