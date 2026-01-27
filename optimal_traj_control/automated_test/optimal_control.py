import sympy as sp
import sympy.diffgeom as dg
import cvxpy as cp

import math

from sympy.core.symbol import Str
from sympy.core.function import AppliedUndef
from cvxpy.expressions.leaf import Leaf
from collections import OrderedDict

from typing import Union, OrderedDict, Optional, List, Tuple


# custom types to reduce complexity
target_sympy_type = Union[
    sp.Symbol, Str, dg.BaseScalarField, sp.Function, sp.Derivative
]
result_cvxpy_sym_type = Union[cp.Variable, cp.Parameter]
sympy_map_type = OrderedDict[target_sympy_type, result_cvxpy_sym_type]


def _recursive_expr_to_cvxpy(
    expr: Union[sp.Expr, Str],
    sympy_map: OrderedDict[
        Union[Str, dg.BaseScalarField], Union[cp.Variable, cp.Parameter]
    ],
) -> Union[cp.Expression, float]:
    if expr.is_number:
        # directly converts this number
        return float(expr)
    if expr in sympy_map:
        return sympy_map[expr]
    elif (
        isinstance(expr, sp.Symbol)
        or isinstance(expr, Str)
        or isinstance(expr, dg.BaseScalarField)
        or isinstance(expr, sp.Function)
        or isinstance(expr, sp.Derivative)
    ):
        # if the expression does not
        param = _create_leaf(expr, sympy_map, cvxpy_type=cp.Parameter)
        sympy_map.update({expr: param})

        return param
    else:
        # handles SymPy expressions
        cvxpy_args = (_recursive_expr_to_cvxpy(arg, sympy_map) for arg in expr.args)

        if isinstance(expr, sp.Add):
            return sum(cvxpy_args)
        elif isinstance(expr, sp.Mul):
            return math.prod(cvxpy_args)
        elif isinstance(expr, sp.Pow):
            base, exp = cvxpy_args
            return cp.power(base, exp)
        elif isinstance(expr, sp.LessThan):
            base, exp = cvxpy_args
            return base <= exp
        else:
            raise NotImplementedError(f"Unsupported SymPy expression: {type(expr)}")


def _create_leaf(
    expr: Union[Str, dg.BaseScalarField, sp.Function],
    shared_lookup: Optional[sympy_map_type] = None,
    cvxpy_type: Leaf = cp.Parameter,
) -> Union[cp.Variable, cp.Parameter]:
    if shared_lookup is not None and expr in shared_lookup:
        # this is if we're re-using the map across multiple generated equations
        return shared_lookup[expr]
    else:
        # create the specified leaf node
        if (
            isinstance(expr, sp.Symbol)
            or isinstance(expr, Str)
            or isinstance(expr, sp.Function)
            or isinstance(expr, sp.Derivative)
        ):
            return cvxpy_type(
                name=sp.latex(expr),
                # hacky solution (figure out better one later)
                nonneg=sp.latex(expr) == "p",
            )  # shares the same name access
        elif isinstance(expr, dg.BaseScalarField):
            coord_sys: dg.CoordSystem = expr.coord_sys
            return cvxpy_type(name=coord_sys.symbols[expr.index].name)
        else:
            raise NotImplementedError(
                f"Unsupported SymPy type for leaf creation: {type(expr)}"
            )


def _convert_sympy_to_cvxpy(
    expr: sp.Expr,
    sympy_vars: List[target_sympy_type],
    shared_sympy_map: Optional[sympy_map_type] = None,
) -> Tuple[cp.Expression, sympy_map_type]:
    sympy_map_vars = {
        var: _create_leaf(var, shared_lookup=shared_sympy_map, cvxpy_type=cp.Variable)
        for var in sympy_vars
    }
    sympy_map = OrderedDict() if shared_sympy_map is None else shared_sympy_map
    sympy_map.update(sympy_map_vars)

    return (_recursive_expr_to_cvxpy(expr, sympy_map), sympy_map)


def _separate_sympy_to_cvxpy_map(
    sympy_map: sympy_map_type,
) -> Tuple[
    OrderedDict[sympy_map_type, cp.Variable], OrderedDict[sympy_map_type, cp.Parameter]
]:

    var_map = OrderedDict(
        item for item in sympy_map.items() if isinstance(item[1], cp.Variable)
    )
    param_map = OrderedDict(
        item for item in sympy_map.items() if isinstance(item[1], cp.Parameter)
    )

    return (var_map, param_map)


def dynamic_unicycle_optimal_traj_derivations():
    # defines the state manifold
    state_mfld = dg.Manifold("M", 5)
    state_mfld_patch = dg.Patch("P", state_mfld)

    x, y, theta, v, omega = sp.symbols(r"x,y,\theta,v,\omega", real=True)

    state_mfld_coords = dg.CoordSystem(
        "StateSpace", state_mfld_patch, (x, y, theta, v, omega)
    )

    (x_sc, y_sc, theta_sc, v_sc, omega_sc) = state_mfld_coords.base_scalars()
    (x_vec, y_vec, theta_vec, v_vec, omega_vec) = state_mfld_coords.base_vectors()

    # define the vector fields of the system

    f = (
        v_sc * sp.cos(theta_sc) * x_vec
        + v_sc * sp.sin(theta_sc) * y_vec
        + omega_sc * theta_vec
    )

    f1 = v_vec
    f2 = omega_vec

    # define the variables associated with tthe trajectory

    t = sp.symbols("t", real=True)
    alpha, beta = sp.symbols(r"\alpha,\beta", real=True)

    x_tr = sp.Function("x_tr")(t)
    y_tr = sp.Function("y_tr")(t)
    theta_tr = sp.Function(r"\theta_tr")(t)
    # v_tr = sp.Function("v_tr")(t)
    # omega_tr = sp.Function(r"\omega_tr")(t)

    # define the constraint to enforce decreasing error
    v_fn = (
        0.5 * alpha * ((x_sc - x_tr) ** 2 + (y_sc - y_tr) ** 2)
        + 0.5 * beta * (theta_sc - theta_tr) ** 2
    )

    Lf_Lf_v = dg.LieDerivative(f, dg.LieDerivative(f, v_fn))
    Lf1_Lf_v = dg.LieDerivative(f1, dg.LieDerivative(f, v_fn))
    Lf2_Lf_v = dg.LieDerivative(f2, dg.LieDerivative(f, v_fn))
    dv_dt = sp.diff(v_fn, t)

    # define the inputs
    u_f, u_t = sp.symbols("f,tau", real=True)

    # define the optimization problem

    p, delta = sp.symbols(r"p,delta", real=True)
    prob_f = 0.5 * (u_f**2 + u_t**2) + p * delta**2
    prob_g = Lf_Lf_v + Lf1_Lf_v * u_f + Lf2_Lf_v * u_t + dv_dt <= delta
    prob_vars = [u_f, u_t, delta]

    return (prob_f, [prob_g], prob_vars)


def convert_sym_prob_to_cvxpy_prob(sympy_f, sympy_gs, sympy_opt_vars) -> Tuple[
    cp.Problem,
    List[Tuple[target_sympy_type, cp.Variable]],
    List[Tuple[target_sympy_type, cp.Parameter]],
]:
    # converts the optimization problem presented as a sympy formulation and
    # converts it into a cvxpy optimization problem
    prob_f_cvxpy, sympy_to_cvxpy_map = _convert_sympy_to_cvxpy(sympy_f, sympy_opt_vars)
    prob_gs_cvxpy = [
        _convert_sympy_to_cvxpy(sympy_g, sympy_opt_vars, sympy_to_cvxpy_map)[0]
        for sympy_g in sympy_gs
    ]

    var_map, param_map = _separate_sympy_to_cvxpy_map(sympy_to_cvxpy_map)

    cvxpy_prob = cp.Problem(cp.Minimize(prob_f_cvxpy), prob_gs_cvxpy)

    var_map_indexed = [tuple(item) for item in var_map.items()]
    param_map_indexed = [tuple(item) for item in param_map.items()]

    return (cvxpy_prob, var_map_indexed, param_map_indexed)


def main():
    (sympy_f, sympy_gs, sympy_opt_vars) = dynamic_unicycle_optimal_traj_derivations()
    (cntrl_prob_cvxpy, prob_var_map, prob_param_map) = convert_sym_prob_to_cvxpy_prob(
        sympy_f, sympy_gs, sympy_opt_vars
    )

    print(cntrl_prob_cvxpy)  # inspect manually to ensure it works
    print(cntrl_prob_cvxpy.get_problem_data(cp.OSQP)[0])

    print()

    print(list(param.name() for label, param in prob_param_map))


if __name__ == "__main__":
    main()
