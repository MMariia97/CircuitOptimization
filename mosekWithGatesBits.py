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
        return list(a)

def findHamiltonian(xx, nvars):
    ### Hamiltonian
    h_dict = {i: xx[i] for i in range(nvars)}
    print("h", h_dict)
    J_dict = {}
    for i in range(nvars-1):
        for j in range(i+1,nvars):
            J_dict[(i, j)] = xx[i*nvars+j+nvars-1]
    print("J", J_dict)

    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_ising(h_dict, J_dict, num_reads=100)
    print("sampleset\n", sampleset)

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
    
    nvars = nvar+bitCount

    # Create a task object
    with mosek.Task() as task:
        # Attach a log stream printer to the task
        task.set_Stream(mosek.streamtype.log, streamprinter)

        # Bound keys for constraints
        bkc = []
        for i in range(len(eq_arr)):
            if (eq_arr[i][(nvars+nvars**2)] == -1):
                bkc.append(mosek.boundkey.lo)
            else:
                bkc.append(mosek.boundkey.fx)

        # Bound values for constraints
        blc = [0] * len(eq_arr)
        buc = []
        for i in range(len(eq_arr)):
            if (eq_arr[i][(nvars+nvars**2)] == -1):
                buc.append(+inf)
            else:
                buc.append(0)

        # Bound keys and values for variables
        bkx = []
        blx = []
        bux=[]
        for i in range(nvars):
            bkx.append(mosek.boundkey.ra)
            blx.append(-2)
            bux.append(2)

        for i in range(nvars**2):
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
        c = [0] * (nvars+nvars**2+2)
        c[nvars**2+nvars] = 1
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
            findHamiltonian(xx, nvars)
            

        elif (solsta == mosek.solsta.dual_infeas_cer or
              solsta == mosek.solsta.prim_infeas_cer):
            print("Primal or dual infeasibility certificate found.\n")
        elif solsta == mosek.solsta.unknown:
            print("Unknown solution status")
        else:
            print("Other solution status")


    


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

