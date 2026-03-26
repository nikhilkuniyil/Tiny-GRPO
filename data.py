from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional


INTEGER_PATTERN = re.compile(r"(?<![\w-])[+-]?\d+")


@dataclass(frozen=True)
class EquationExample:
    # Keep the rendered prompt alongside the structured values so the same
    # object can feed both the model and the reward code.
    prompt: str
    equation: str
    solution: int
    coefficient: int
    bias: int
    rhs: int


def format_linear_equation(coefficient: int, bias: int, rhs: int) -> str:
    if coefficient == 0:
        raise ValueError("Coefficient must be non-zero.")

    if coefficient == 1:
        left_side = "x"
    elif coefficient == -1:
        left_side = "-x"
    else:
        left_side = f"{coefficient}x"

    if bias > 0:
        left_side = f"{left_side} + {bias}"
    elif bias < 0:
        left_side = f"{left_side} - {abs(bias)}"

    return f"{left_side} = {rhs}"


def solve_linear_equation(coefficient: int, bias: int, rhs: int) -> int:
    if coefficient == 0:
        raise ValueError("Coefficient must be non-zero.")

    numerator = rhs - bias
    if numerator % coefficient != 0:
        raise ValueError("Equation does not have an integer solution.")
    return numerator // coefficient


def generate_equation_example(
    *,
    min_solution: int = -10,
    max_solution: int = 10,
    min_coefficient: int = -5,
    max_coefficient: int = 5,
    min_bias: int = -10,
    max_bias: int = 10,
    rng: Optional[random.Random] = None,
) -> EquationExample:
    rng = rng or random.Random()

    coefficient = 0
    while coefficient == 0:
        # Avoid zero so every example is a real linear equation in x.
        coefficient = rng.randint(min_coefficient, max_coefficient)

    solution = rng.randint(min_solution, max_solution)
    bias = rng.randint(min_bias, max_bias)
    # Choose the solution first, then construct the right-hand side so the
    # final equation is guaranteed to have an integer answer.
    rhs = coefficient * solution + bias
    equation = format_linear_equation(coefficient, bias, rhs)

    return EquationExample(
        prompt=f"Solve for x: {equation}",
        equation=equation,
        solution=solution,
        coefficient=coefficient,
        bias=bias,
        rhs=rhs,
    )


def generate_dataset(num_examples: int, *, seed: int = 42) -> list[EquationExample]:
    rng = random.Random(seed)
    return [generate_equation_example(rng=rng) for _ in range(num_examples)]


def extract_first_integer(text: str) -> Optional[int]:
    # GRPO should reward the answer, not exact formatting, so we accept
    # responses like "x = 5" or "The answer is -3".
    match = INTEGER_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group())


def exact_match_reward(response_text: str, target_answer: int) -> float:
    # v1 reward is intentionally binary and deterministic.
    predicted_answer = extract_first_integer(response_text)
    return 1.0 if predicted_answer == target_answer else 0.0
