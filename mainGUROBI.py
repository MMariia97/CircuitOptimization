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
from gurobipy import *

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


def addCNOT1ToExpr(rows, cols):
    expr1 = ''
    # CNOT with 1st control qubit
    arr1 = np.zeros((rows, cols))
    i = 0
    for st in range(len(states)):
        res = np.matmul(CNOT_matrix, list(states.values())[st])
        l = list(states.keys())[st].split()
        arr1[i] = concat(l, [key for key, val in states.items()
                         if np.allclose(val, res)][0].split())
        i = i+1

    for row in range(rows):
        for col in range(cols):
            xnum = row*rows+col
            xs = 'x%d' % xnum
            if arr1[row][col] == 0:
                expr1 += '~%s & ' % xs
            else:
                expr1 += '%s & ' % xs

    # find len of suffex
    andstr = len(' & ')
    # slice out from string
    expr1 = expr1[:-andstr]
    return expr1


def addCNOT2ToExpr(rows, cols):
    expr1 = ''
    # CNOT with 2nd control qubit
    arr2 = np.zeros((rows, cols))
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
        arr2[i] = concat(l, resKey.split())
        i = i+1

    for row in range(rows):
        for col in range(cols):
            xnum = row*rows+col
            xs = 'x%d' % xnum
            if arr2[row][col] == 0:
                expr1 += '~%s & ' % xs
            else:
                expr1 += '%s & ' % xs

    # find len of suffex
    andstr = len(' & ')
    # slice out from string
    expr1 = expr1[:-andstr]
    return expr1


def addNOT1ToExpr(rows, cols):
    expr1 = ''
    # NOT for the 1st qubit
    arr3 = np.zeros((rows, cols))
    i = 0
    for st in range(len(states)):
        l = list(states.keys())[st].split()
        arr3[i] = concat(l, [int(not bool(int(l[0]))), l[1]])
        i = i+1

    for row in range(rows):
        for col in range(cols):
            xnum = row*rows+col
            xs = 'x%d' % xnum
            if arr3[row][col] == 0:
                expr1 += '~%s & ' % xs
            else:
                expr1 += '%s & ' % xs

    # find len of suffex
    andstr = len(' & ')
    # slice out from string
    expr1 = expr1[:-andstr]
    return expr1


def addNOT2ToExpr(rows, cols):
    expr1 = ''
    # NOT for the 2nd qubit
    arr4 = np.zeros((rows, cols))
    i = 0
    for st in range(len(states)):
        res = np.matmul(CNOT_matrix, list(states.values())[st])
        l = list(states.keys())[st].split()
        arr4[i] = concat(l, [l[0], int(not bool(int(l[1])))])
        i = i+1

    for row in range(rows):
        for col in range(cols):
            xnum = row*rows+col
            xs = 'x%d' % xnum
            if arr4[row][col] == 0:
                expr1 += '~%s & ' % xs
            else:
                expr1 += '%s & ' % xs
    return expr1

def convert_to_minus(x):
  # 0 -> 1, 1 -> -1
  return (1 - 2*numpy.multiply(x, 1))

def getEquations(*x, symexpr, N):
  a = numpy.zeros(N*(N+1)+2)
  for i in range(N):
    a[i] = convert_to_minus(x[i]) # find coefficients for h_i
    for row in range(N):
      for col in range(N):
        if (row < col):
          a[col + row*N + N] = convert_to_minus(x[row])*convert_to_minus(x[col]) # find coefficients for J_ij, i<j
        else:
          a[col + row*N + N] = 0
    sat = symexpr(x)
    if (sat == True):
      a[N*(N+1)] = 0
      print("TRUE!!!")
    else:
      a[N*(N+1)] = -1 # find coefficients for g
    a[N*(N+1)+1] = -1 # coefficient for k
    return a

def findHamiltonian(N, eq_arr):
    # Create a new model
    m = Model()

    # Create variables
    h = m.addVars(range(0, N), lb=-2, ub=2, name="h")
    JJ = m.addVars(range(N, N**2+N), lb=-1, ub=1, name="JJ")
    g = m.addVars(range(N+N**2, N+N**2+1), name="g")
    k = m.addVars(range(N+N**2+1, N+N**2+2), lb=-GRB.INFINITY, name="k")
    # Set objective function
    m.setObjective(g[N+N**2], GRB.MAXIMIZE)
    print("h", h)
    print("JJ", JJ)
    print("g", g)
    print("k", k)
    total_dict = {**h, **JJ, **g, **k}
    print("total_dict", list(total_dict.values()))
    prod = numpy.matmul(eq_arr, list(total_dict.values()))
    print("prod", prod)
    #Add constraints
    for i in range(len(prod)):
        if (eq_arr[i][(N+N**2)] == -1):
            #  print(LpConstraint(prod[i], rhs=0, sense=1))
            m.addConstr(prod[i] >= 0, 'c%d'% i)
        else:
            # print(LpConstraint(prod[i], rhs=0, sense=0))
             m.addConstr(prod[i] == 0, 'c%d'% i)

    # Optimize model
    m.optimize()

    #Print values for decision variables
    J_dict = {}
    h_dict = {}
    for v in h.values():
        print("h {}: {}".format(v.varName, v.x))
        h_dict[i] = v.x
    for v in JJ.values():
        print("JJ {}: {}".format(v.varName, v.x))
        J_dict[i] = v.x
    for v in h.values():
        print("{}: {}".format(v.varName, v.x))
    for v in g.values():
        print("g {}: {}".format(v.varName, v.x))
    for v in k.values():
        print("k {}: {}".format(v.varName, v.x))

    #for v in m.getVars():
    #    print(v.varName)
    return h_dict, J_dict


def main():
    with open('inputTruthTable.txt', 'r') as f:
        truthTable = [[int(num) for num in line.split(',')] for line in f]
    rows = len(truthTable)
    cols = len(truthTable[0])
    nvar = rows*cols
    symbols('x0:%d' % nvar)

    ### construct Boolean expression
    expr1 = getExprFromTruthTable(truthTable, rows, cols) + ' & (' + addCNOT1ToExpr(
        rows, cols) + ' | ' + addCNOT2ToExpr(rows, cols) + ' | ' + addNOT1ToExpr(rows, cols) + ' | ' + addNOT2ToExpr(rows, cols)
    # find len of suffex
    andstr = len(' & ')
    # slice out from string
    expr1 = expr1[:-andstr]
    expr1 += ')'
    print(expr1)
    ### cnf
    bs = expr(str(expr1))
    symexpr = lambdify([symbols('x0:%d' % nvar)],sympify(str(bs.to_cnf())))
    print(sympify(str(bs.to_cnf())))
    ### LP problem
    num_cores = cpu_count()
    pool = Pool(num_cores)
    partfunc = functools.partial(getEquations, symexpr=symexpr, N=nvar)
    res = pool.starmap_async(partfunc, list(itertools.product([False, True], repeat=nvar)), chunksize=10)
    pool.close()
    pool.join()
    eq_arr = res.get()

    h_dict, J_dict = findHamiltonian(nvar, eq_arr)

    ### QA
 #   sampler = neal.SimulatedAnnealingSampler()

  #  sampleset = sampler.sample_ising(h_dict, J_dict, num_reads=10)
   # print("sampleset\n", sampleset)

   # print("\nto qubo")
    #print(dimod.ising_to_qubo(h_dict, J_dict))
    return 0


if __name__ == '__main__':
    main()
