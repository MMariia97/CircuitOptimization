from operator import concat
from sympy import *
from sympy.logic.boolalg import to_cnf
import numpy as np
from sympy.abc import *
import fileinput
from pyeda.inter import *
import pyeda
import itertools
import argparse
import pyqubo
from pyqubo import Spin
import numpy
import sympy as sym
import math
from pulp import *
import neal
import dimod
import json
import functools
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import cpu_count

CNOT_matrix = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]])

qubit0 = np.array([[1], [0]])
qubit1 = np.array([[0], [1]])
states = dict()
states['0 0'] = np.kron(qubit0, qubit0)
states['0 1'] = np.kron(qubit0, qubit1)
states['1 0'] = np.kron(qubit1, qubit0)
states['1 1'] = np.kron(qubit1, qubit1)


def convert_to_minus(x):
    # 0 -> 1, 1 -> -1
    return (1 - 2*numpy.multiply(x, 1))


def getEquations(*x, symexpr, N):
    a = numpy.zeros(N*(N+1)+2)
    for i in range(N):
        a[i] = convert_to_minus(x[i])  # find coefficients for h_i
        for row in range(N):
            for col in range(N):
                if (row < col):
                    # find coefficients for J_ij, i<j
                    a[col + row*N +
                        N] = convert_to_minus(x[row])*convert_to_minus(x[col])
                else:
                    a[col + row*N + N] = 0
        sat = symexpr(x)
        print(sat)
        if (sat == True):
            a[N*(N+1)] = 0
        else:
            a[N*(N+1)] = -1  # find coefficients for g
        a[N*(N+1)+1] = -1  # coefficient for k
        return a

def addToExpr(arr, circBitsArr, rows, cols):
    expr1 = ''
    for row in range(rows):
        for col in range(cols):
            xnum = row*rows+col
            xs = 'x%d' % xnum
            if arr[row][col] == 0:
                expr1 += '~%s & ' % xs
            else:
                expr1 += '%s & ' % xs

    for i in range(len(circBitsArr)):
        xnum = rows*cols + i
        xs = 'x%d' % xnum
        if int(circBitsArr[i]) == 0:
            expr1 += '~%s & ' % xs
        else:
            expr1 += '%s & ' % xs
    # find len of suffex
    andstr = len(' & ')
    # slice out from string
    expr1 = expr1[:-andstr]
    return expr1

def getExprFromTruthTable(truthTable, rows, cols):
    expr1 = ''
    for row in range(rows):
        for col in range(cols):
            xnum = row*rows+col
            xs = 'x%d' % xnum
            if truthTable[row][col] == 0:
                expr1 += '~%s & ' % xs
            else:
                expr1 += '%s & ' % xs
    # find len of suffex
    andstr = len(' & ')
    # slice out from string
    expr1 = expr1[:-andstr]
    return expr1


def addCNOT1ToExpr(rows, cols, circBitsArr):
    # CNOT with 1st control qubit
    arr = np.zeros((rows, cols))
    i = 0
    for st in range(len(states)):
        res = np.matmul(CNOT_matrix, list(states.values())[st])
        l = list(states.keys())[st].split()
        arr[i] = concat(l, [key for key, val in states.items()
                         if np.allclose(val, res)][0].split())
        i = i+1

    expr1 = addToExpr(arr, circBitsArr, rows, cols)
    return expr1


def addCNOT2ToExpr(rows, cols, circBitsArr):
    # CNOT with 2nd control qubit
    arr = np.zeros((rows, cols))
    i = 0
    for st in range(len(states)):
        revStKey = ' '.join(
            list(list(reversed(list(states.keys())[st].split()))))
        revStVal = [val for key, val in states.items() if key == revStKey]
        res = np.matmul(CNOT_matrix, list(revStVal))
        revResKey = [key for key, val in states.items(
        ) if np.allclose(val, res)][0].split()
        resKey = ' '.join(list(list(reversed(list(revResKey)))))
        l = list(states.keys())[st].split()
        arr[i] = concat(l, resKey.split())
        i = i+1

    expr1 = addToExpr(arr, circBitsArr, rows, cols)
    return expr1


def addNOT1ToExpr(rows, cols, circBitsArr):
    # NOT for the 1st qubit
    arr = np.zeros((rows, cols))
    i = 0
    for st in range(len(states)):
        l = list(states.keys())[st].split()
        arr[i] = concat(l, [int(not bool(int(l[0]))), l[1]])
        i = i+1

    expr1 = addToExpr(arr, circBitsArr, rows, cols)
    return expr1


def addNOT2ToExpr(rows, cols, circBitsArr):
    # NOT for the 2nd qubit
    arr = np.zeros((rows, cols))
    i = 0
    for st in range(len(states)):
        l = list(states.keys())[st].split()
        arr[i] = concat(l, [l[0], int(not bool(int(l[1])))])
        i = i+1

    expr1 = addToExpr(arr, circBitsArr, rows, cols)
    return expr1


def convert_to_minus(x):
    # 0 -> 1, 1 -> -1
    return (1 - 2*numpy.multiply(x, 1))


def getEquations(*x, symexpr, N):
    a = numpy.zeros(N*(N+1)+2)
    for i in range(N):
        a[i] = convert_to_minus(x[i])  # find coefficients for h_i
        for row in range(N):
            for col in range(N):
                if (row < col):
                    # find coefficients for J_ij, i<j
                    a[col + row*N +
                        N] = convert_to_minus(x[row])*convert_to_minus(x[col])
                else:
                    a[col + row*N + N] = 0
        sat = symexpr(x)
        if (sat == True):
            a[N*(N+1)] = 0
        else:
            a[N*(N+1)] = -1  # find coefficients for g
        a[N*(N+1)+1] = -1  # coefficient for k
        return a


def findHamiltonian(N, eq_arr):
    print("in findHamiltonian func")
    prob = LpProblem("Hamiltonian_coeff", LpMaximize)
    h = LpVariable.dicts("h", range(0, N), lowBound=-2, upBound=2)
    JJ = LpVariable.dicts("JJ", range(N, N**2+N), lowBound=-1, upBound=1)
    g = LpVariable.dicts("g", range(N+N**2, N+N**2+1), lowBound=0)
    k = LpVariable.dicts("k", range(N+N**2+1, N+N**2+2))
    prob += g[N+N**2]
    total_dict = {**h, **JJ, **g, **k}  # variables
    var_list = list(total_dict.values())
 #   print("var_list", var_list)
    prod = numpy.matmul(eq_arr, var_list)
   # print("prod", prod)
 #   print("constraints")
    for i in range(len(prod)):
        if (eq_arr[i][(N+N**2)] == -1):
   #         print(LpConstraint(prod[i], rhs=0, sense=1))
            prob += LpConstraint(prod[i], rhs=0, sense=1)
        else:
   #         print(LpConstraint(prod[i], rhs=0, sense=0))
            prob += LpConstraint(prod[i], rhs=0, sense=0)
    status = prob.solve(CPLEX_CMD)

    J_dict = {}
    h_dict = {}
    print("status\n")
    print(LpStatus[status])
    print("\nh_dict\n")
    for i in (range(len(h))):
        if (h[i].varValue != None):
            h_dict[i] = h[i].varValue
        else:
            h_dict[i] = 0
    print(h_dict)
    print("\nJ_dict\n")
    for i in range(N):
        for j in range(N):
            if (i < j):
                J_dict[(i, j)] = JJ[i*N+j+N].varValue
    print(J_dict)
    print("\ng\n")
    print(g[N+N**2].varValue)
    print("\nk\n")
    print(k[N+N**2+1].varValue)
    return h_dict, J_dict


def main():
    print(listSolvers(onlyAvailable=True))
    with open('inputTruthTable.txt', 'r') as f:
        truthTable = [[int(num) for num in line.split(',')] for line in f]
    rows = len(truthTable)
    cols = len(truthTable[0])
    nvar = rows*cols
    symbols('x0:%d' % nvar)

    circCount = 4
    circArr = [addCNOT1ToExpr, addCNOT2ToExpr, addNOT1ToExpr, addNOT2ToExpr]
    bitCount = math.ceil(math.log2(circCount))
    # construct Boolean expression
    expr1 = getExprFromTruthTable(truthTable, rows, cols) + ' & ('
    for i in range(len(circArr)):
        expr1 += circArr[i](rows, cols, list(format(i, 'b').zfill(bitCount)))
        expr1 += ' | '

    # find len of suffex
    andstr = len(' | ')
    # slice out from string
    expr1 = expr1[:-andstr]

    expr1 += ')'
    print('expression ', expr1)
    # cnf
    bs = expr(str(expr1))
    cnf = sympify(str(bs.to_cnf()))
    symexpr = lambdify([symbols('x0:%d' % (nvar+bitCount))],cnf)
    print('cnf ', cnf)

    ### LP problem
    num_cores = cpu_count()
    pool = Pool(num_cores)
    partfunc = functools.partial(getEquations, symexpr=symexpr, N=(nvar+bitCount))
    res = pool.starmap_async(partfunc, list(itertools.product([False, True], repeat=(nvar+bitCount))), chunksize=10)
    pool.close()
    pool.join()
    eq_arr = res.get()

    print("after starmap")
    h_dict, J_dict = findHamiltonian((nvar+bitCount), eq_arr)

    ### QA
    sampler = neal.SimulatedAnnealingSampler()

    sampleset = sampler.sample_ising(h_dict, J_dict, num_reads=1)
    print("sampleset\n", sampleset)

    print("\nto qubo")
    print(dimod.ising_to_qubo(h_dict, J_dict))

    return 0


if __name__ == '__main__':
    main()
