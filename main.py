from sympy import *
from sympy.logic.boolalg import to_cnf

from sympy.abc import *
import fileinput
from pyeda.inter import *
import pyeda


def main():
    #### sympy -- very slow
  #  for line in fileinput.input(files ='input.txt'):
      #  f = lambdify([A, B, C],to_cnf(line))
    #    print(to_cnf(line, simplify=True, force=True))

    ##### pyeda -- quickly

 #   symbols('a0,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r')
    for line in fileinput.input(files ='input.txt'):
        # pyeda.boolalg.expr.ConjNormalForm(16, line)
        print(type(line))
        bs = expr(line)
        print(bs.to_cnf()) # this works
    # it works
     #   l, cnf = pyeda.boolalg.expr.expr2dimacscnf(bs.to_cnf()) # set of clauses
     #   print(cnf)

      #  print(list(bs.to_cnf().satisfy_all())) # it works

    return 0



if __name__ == '__main__':
    main()
