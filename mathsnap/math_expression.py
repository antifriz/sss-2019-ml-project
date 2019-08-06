class MathExpression:
    latex: str


def math_expression_from_latex(latex: str):
    math_expression = MathExpression()
    math_expression.latex = latex
    return math_expression
