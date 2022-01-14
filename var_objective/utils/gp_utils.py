import sympy
import numpy as np

import sys
import os
import re
from .generator import Generator

def mask_X(s):
    s = s.replace('X0', 'O')
    s = s.replace('X1', 'P')
    s = s.replace('X2', 'Q')
    s = s.replace('X3', 'R')
    return s

def back_X(s):
    s = s.replace('O', 'X0')
    s = s.replace('P', 'X1')
    s = s.replace('Q', 'X2')
    s = s.replace('R', 'X3')
    return s

def get_var_pos():
    X0 = sympy.Symbol('X0', positive=True)
    X1 = sympy.Symbol('X1', positive=True)
    X2 = sympy.Symbol('X2', positive=True)
    X3 = sympy.Symbol('X3', positive=True)
    C = sympy.Symbol('C', positive=True)

    VarDict = {
        'X0': X0,
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'C': C,
    }
    return VarDict

def gp_to_pysym_with_coef(est_gp, tol=None, tol2=None, expand=False):
    VarDict = get_var_pos()
    f_star = est_gp._program
    f_star_list, var_list, coef_list = parse_program_to_list(f_star.program)
    f_star_infix = Generator.prefix_to_infix(f_star_list, variables=var_list, coefficients=coef_list)
    f_star_infix2 = f_star_infix.replace('{', '').replace('}', '')
    if f_star_infix2 == f_star_infix:
        f_star_sympy = Generator.infix_to_sympy(f_star_infix, VarDict, "simplify")
        return (f_star_sympy, f_star_sympy)

    f_star_sympy = Generator.infix_to_sympy(f_star_infix2, VarDict, "simplify")

    if expand:
        f_star_sympy = sympy.expand(f_star_sympy)

    fs = str(f_star_sympy)
    print(fs)
    f_star_sympy_coeff = f_star_sympy

    fs = mask_X(fs)
    if tol is None:
        fs = re.sub(r'([0-9]*\.[0-9]+|[0-9]+)', 'C', fs)
    else:
        consts = re.findall(r'([0-9]*\.[0-9]+|[0-9]+)', fs)
        for const in consts:
            if const in ('1', '2', '3', '4', '5', '6', '7', '8', '9'):
                continue
            if (float(const) < 1 + tol) and (float(const) > 1 - tol):
                fs = fs.replace(const, '1')
            elif (tol2 is not None) and (float(const) < tol2) and (float(const) > -1 * tol2):
                fs = fs.replace(const, '0')
            else:
                fs = fs.replace(const, 'C')

    fs = back_X(fs)
    print(fs)
    f_star_sympy = Generator.infix_to_sympy(fs, VarDict, "simplify")
    return (f_star_sympy_coeff, f_star_sympy)


def check_equal(f1, f2):
    return sympy.simplify(f1 - f2) == 0


def parse_program_to_list(program):
    symbol_list = list()
    var_list = list()
    coef_list = list()

    for i in program:
        if isinstance(i, int):
            symbol_list.append('X' + str(i))
            var_list.append('X' + str(i))
        elif isinstance(i, float):
            symbol_list.append(str(i))
            coef_list.append(str(i))
        else:
            if i.name == 'log':
                symbol_list.append('ln')
            elif i.name == 'neg':
                symbol_list.append('sub')
                symbol_list.append('0')
            else:
                symbol_list.append(i.name)

    var_list = list(set(var_list))
    coef_list = list(set(coef_list))
    return symbol_list, var_list, coef_list