from typing import NamedTuple

from mathsnap.math_expression import MathExpression, math_expression_from_latex


class SolverResult(NamedTuple):
    solution: MathExpression


class Solver:
    def process(self, problem: MathExpression) -> SolverResult:
        raise NotImplementedError()


class DummySolver(Solver):
    def process(self, problem: MathExpression) -> SolverResult:
        return SolverResult(
            solution=math_expression_from_latex('42')
        )


class EvalSolver(Solver):
    def process(self, problem: MathExpression) -> SolverResult:
        try:
            return SolverResult(solution=math_expression_from_latex(eval(problem.latex)))
        except SyntaxError:
            return SolverResult(solution=math_expression_from_latex("Syntax Error"))
        except:
            return SolverResult(solution=math_expression_from_latex("Error"))
