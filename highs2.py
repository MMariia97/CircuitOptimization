from operator import concat
from sympy import *
from sympy.logic.boolalg import to_cnf
import numpy as np
from sympy.abc import *
from pyeda.inter import *
import itertools
from pyqubo import Spin
import numpy
import sympy as sym
import math
import neal
import dimod
import json
import functools
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import cpu_count
import mosek
from scipy import sparse
import sys
import highspy

# Since the value of infinity is ignored, we define it solely
# for symbolic purposes
inf = highspy.kHighsInf

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
    return (1.0 - 2.0*numpy.multiply(x, 1.0))

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

def getEquations(*x, symexpr, N):
    a = [0] * (int(N+N*(N-1)/2+2))
    for i in range(N):
        a[i] = convert_to_minus(x[i])  # find coefficients for h_i
    for row in range(N):
        for col in range(N):
            if (row < col):
                # find coefficients for J_ij, i<j
                a.append(convert_to_minus(x[row])*convert_to_minus(x[col]))
    sat = symexpr(x)
    if (sat == True):
        a.append(0.0)
    else:
        a.append(-1.0)  # find coefficient for g
    a.append(-1.0)  # coefficient for k

    return list(a)

def findHamiltonian(xx, nvars):
    ### Hamiltonian
    h_dict = {i: xx[i] for i in range(nvars)}
    J_dict = {}

    i = 0
    for j in range(nvars-1):
        for k in range(j+1,nvars):
            J_dict[(j, k)] = xx[i+nvars]
            i=i+1

    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_ising(h_dict, J_dict, num_reads=100)
    print("sampleset\n", sampleset)

def solveLinearProblem(nvars, eq_arr):
    h = highspy.Highs()
    lp = highspy.HighsLp()
    lp.num_col_ = int(nvars+nvars*(nvars-1)/2+2)
    lp.num_row_ = 2**nvars
    lp.sense_ = highspy.ObjSense.kMaximize
    c = [0] * (nvars+int(nvars*(nvars-1)/2)+2)
    c[int(nvars+nvars*(nvars-1)/2)] = 1.0
    lp.col_cost_ = np.array(c, dtype=np.double)
    lp.col_lower_ = np.array(list([-2] * nvars + [-1] * (int(nvars*(nvars-1)/2)) + [0] + [-inf]), dtype=np.double)
    lp.col_upper_ = np.array(list([2] * nvars + [1] * (int(nvars*(nvars-1)/2)) + [inf] + [inf]), dtype=np.double)
    lp.row_lower_ = np.array([0] * (2**nvars), dtype=np.double)
  #  buc = []
   # for i in range(len(eq_arr)):
   #     if (eq_arr[i][(nvars+int(nvars*(nvars-1)/2))] == -1.0):
   #         buc.append(+inf)
   #     else:
   #         buc.append(0)
   # lp.row_upper_ = np.array(buc, dtype=np.double)
    lp.row_upper_ = np.array([0] * (2**nvars), dtype=np.double)
    eq_arr_index = []
    eq_arr_start = [0]
    eq_arr_val = []
    for eq in eq_arr:
        indNonZero = numpy.flatnonzero(eq)
        eq_arr_index = eq_arr_index+list(indNonZero)
        eq_arr_start.append(numpy.count_nonzero(eq)+eq_arr_start[-1])
        eq_arr_val = eq_arr_val+[eq[index] for index in indNonZero]
    lp.a_matrix_.start_ = eq_arr_start
    lp.a_matrix_.index_ = eq_arr_index 
    lp.a_matrix_.value_ = np.array(eq_arr_val, dtype=np.double)
    h.passModel(lp)
    h.run()
    solution = h.getSolution()
    basis = h.getBasis()
    info = h.getInfo()
    model_status = h.getModelStatus()
    print('Model status = ', h.modelStatusToString(model_status))
    print('Optimal objective = ', info.objective_function_value)
    print('Iteration count = ', info.simplex_iteration_count)
    print('Primal solution status = ', h.solutionStatusToString(info.primal_solution_status))
    print('Dual solution status = ', h.solutionStatusToString(info.dual_solution_status))
    print('Basis validity = ', h.basisValidityToString(info.basis_validity))
    num_var = h.getNumCol()
    num_row = h.getNumRow()
    print('Variables')
    for icol in range(num_var):
        print(icol, solution.col_value[icol], h.basisStatusToString(basis.col_status[icol]))
  #  print('Constraints')
   # for irow in range(num_row):
   #   print(irow, solution.row_value[irow], h.basisStatusToString(basis.row_status[irow]))
    
    findHamiltonian(solution.col_value, nvars)
    # Clear so that incumbent model is empty
    h.clear()


def main():
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

    solveLinearProblem(nvar+bitCount, eq_arr)

    return 0


# call the main function
try:
    main()
except mosek.Error as e:
    print("ERROR: %s" % str(e.errno))
    if e.msg is not None:
        print("\t%s" % e.msg)
        sys.exit(1)
except:
    import traceback
    traceback.print_exc()
    sys.exit(1)

