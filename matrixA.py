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
import sys
import mosek
from scipy import sparse

# Since the value of infinity is ignored, we define it solely
# for symbolic purposes
inf = 0.0

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

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

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
    return list(a)

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

    print("eq_arr", eq_arr)
    
    # Create a task object
    with mosek.Task() as task:
        # Attach a log stream printer to the task
        task.set_Stream(mosek.streamtype.log, streamprinter)

        # Bound keys for constraints
        bkc = []
        for i in range(len(eq_arr)):
            if (eq_arr[i][(nvar+nvar**2)] == -1):
                bkc.append(mosek.boundkey.lo)
            else:
                bkc.append(mosek.boundkey.fx)

        # Bound values for constraints
        blc = [0] * len(eq_arr)
        buc = []
        for i in range(len(eq_arr)):
            if (eq_arr[i][(nvar+nvar**2)] == -1):
                buc.append(+inf)
            else:
                buc.append(0)

        # Bound keys and values for variables
        bkx = []
        blx = []
        bux=[]
        for i in range(nvar):
            bkx.append(mosek.boundkey.ra)
            blx.append(-2)
            bux.append(2)

        for i in range(nvar**2):
            bkx.append(mosek.boundkey.ra)
            blx.append(-1)
            bux.append(1)

        bkx.append(mosek.boundkey.lo)
        blx.append(0)
        bux.append(+inf)

        bkx.append(mosek.boundkey.fr)
        blx.append(-inf)
        bux.append(+inf)

        # Objective coefficients
        c = [0] * (nvar+nvar**2+2)
        c[nvar**2+nvar] = 1
        # Below is the sparse representation of the A
        # matrix stored by column.

        numvar = len(bkx)
        numcon = len(bkc)
        asub = []
        aval = []
        I, J, V = sparse.find(eq_arr)
        for j in range(numvar):
            indices = [i for i, x in enumerate(J) if x == j]
            asub.append(list(I[indices]))
            aval.append(list(V[indices]))
          #  print("aval", aval)
        print(asub)
        print(aval)
        # Append 'numcon' empty constraints.
        # The constraints will initially have no bounds.
        task.appendcons(numcon)

        # Append 'numvar' variables.
        # The variables will initially be fixed at zero (x=0).
        task.appendvars(numvar)

        for j in range(numvar):
            # Set the linear term c_j in the objective.
            task.putcj(j, c[j])

            # Set the bounds on variable j
            # blx[j] <= x_j <= bux[j]
            task.putvarbound(j, bkx[j], blx[j], bux[j])

            # Input column j of A
            task.putacol(j,                  # Variable (column) index.
                         asub[j],            # Row index of non-zeros in column j.
                         aval[j])            # Non-zero Values of column j.

        # Set the bounds on constraints.
         # blc[i] <= constraint_i <= buc[i]
        for i in range(numcon):
            task.putconbound(i, bkc[i], blc[i], buc[i])

        # Input the objective sense (minimize/maximize)
        task.putobjsense(mosek.objsense.maximize)

        # Solve the problem
        task.optimize()
        # Print a summary containing information
        # about the solution for debugging purposes
        task.solutionsummary(mosek.streamtype.msg)

        # Get status information about the solution
        solsta = task.getsolsta(mosek.soltype.bas)

        if (solsta == mosek.solsta.optimal):
            xx = task.getxx(mosek.soltype.bas)
            
            print("Optimal solution: ")
            for i in range(numvar):
                print("x[" + str(i) + "]=" + str(xx[i]))
        elif (solsta == mosek.solsta.dual_infeas_cer or
              solsta == mosek.solsta.prim_infeas_cer):
            print("Primal or dual infeasibility certificate found.\n")
        elif solsta == mosek.solsta.unknown:
            print("Unknown solution status")
        else:
            print("Other solution status")

    


    return 0


if __name__ == '__main__':
    main()
