import re
from functools import partial
from sympy import Symbol
# My convenient math parser
# install simpleeval and py_expression_eval
from simpleeval import simple_eval
from py_expression_eval import Parser

# install sympy and sympytorch
from sympy import symbols, lambdify, Add

# dynamic import
import importlib
try:
    importlib.import_module("sympytorch")
    import sympytorch
except ModuleNotFoundError:
    print("sympytorch is not installed, hence its disabled functionality")

# split_by_addition is only used when return_torch=True.
def math_eval(s, basic_vars=None, return_function=True, return_torch=False, split_by_addition=False):
    if basic_vars is None:
        basic_vars = Parser().parse(s).variables()
    basic_vars = sorted(set(basic_vars))
    names = {c:symbols(c) for c in basic_vars}
    out = simple_eval(s, names=names)

    # Some more NNs?
    if return_torch:
        if split_by_addition and out.func == Add:
            return sympytorch.SymPyModule(expressions=list(out.args)), basic_vars
        return sympytorch.SymPyModule(expressions=[out]), basic_vars

    # return sympy function
    if return_function:
        return lambdify(expr=out, args=basic_vars), basic_vars

    return out, basic_vars

# extract function from an equation obtained by pysr package
def extract_function(equation):
    floats = re.findall("\d+\.\d+", equation.equation)

    notaions = [chr(97+i) for i in range(len(floats))]
    ind_notations = Parser().parse(equation.sympy_format).variables()

    new_equation = equation.equation
    for f,n in zip(floats, notaions):
        new_equation = new_equation.replace(f, n)

    function = lambdify(args=list(map(Symbol, ind_notations+notaions)),
                        expr=new_equation)

    return function, ind_notations, dict(zip(notaions, map(float, floats)))
