import os
from typing import Literal

import yaml
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.func import RetryPolicy, entrypoint, task
from pydantic import BaseModel, Field

from src.utils.json_helpers import dump_json, load_json

CONNECTIONS_FILEPATH = "data/connections.jsonl"
PROMPTS_FILEPATH = "prompts/prompts.yml"
STATS_FILEPATH = "output/stats.json"
MAX_TRIES = 3
PROMPT_SETTINGS = yaml.safe_load(open(PROMPTS_FILEPATH))


def custom_retry_on(exc: Exception) -> bool:
    import httpx
    import requests

    if isinstance(exc, ConnectionError):
        return True
    if isinstance(
        exc,
        (
            ValueError,
            TypeError,
            ArithmeticError,
            ImportError,
            LookupError,
            NameError,
            SyntaxError,
            RuntimeError,
            ReferenceError,
            StopIteration,
            StopAsyncIteration,
            OSError,
        ),
    ):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    if isinstance(exc, requests.HTTPError):
        return 500 <= exc.response.status_code < 600 if exc.response else True
    return True


# Define a retry policy
retry_policy = RetryPolicy(
    max_attempts=3,
    initial_interval=0.5,
    backoff_factor=2,
    jitter=True,
    retry_on=custom_retry_on,
)

'''
Connections are stored in a JSONL file. Each line is a puzzle with the following fields:
- words: a list of 16 words
- solution: 
    - groups: 
        - words: a list of 4 words
        - reason: a string explaining why the words are connected

'''


class ValidationError(Exception):
    pass


class Grade(BaseModel):
    grade: Literal["fits", "doesn't fit"] = Field(
        description="Decide if the word fits with the group.",
    )


# Schema for structured output to use in evaluation
class Feedback(BaseModel):
    submit: bool = Field(
        description="True if the solution is correct, False otherwise.",
    )
    advice: str | None = Field(
        description="Advice regarding how to solve this puzzle. State specific changes to make to correct the solution.",
    )


class Solution(BaseModel):
    yellow_group: list[str] = Field(
        description="The 4 words that go best together for the yellow group.",
    )
    yellow_group_reason: str = Field(
        description="The very brief reason why the words go together for the yellow group.",
    )
    green_group: list[str] = Field(
        description="The 4 words that go best together for the green group.",
    )
    green_group_reason: str = Field(
        description="The very brief reason why the words go together for the green group.",
    )
    blue_group: list[str] = Field(
        description="The 4 words that go best together for the blue group.",
    )
    blue_group_reason: str = Field(
        description="The very brief reason why the words go together for the blue group.",
    )
    purple_group: list[str] = Field(
        description="The 4 words that go best together for the purple group.",
    )
    purple_group_reason: str = Field(
        description="The very brief reason why the words go together for the purple group.",
    )

    @property
    def comparable_solution(self):
        return set(
            frozenset(group)
            for group in [
                self.yellow_group,
                self.green_group,
                self.blue_group,
                self.purple_group,
            ]
        )

    @property
    def serializable_solution(self):
        return {
            "yellow_group": list(self.yellow_group),
            "yellow_group_reason": self.yellow_group_reason,
            "green_group": list(self.green_group),
            "green_group_reason": self.green_group_reason,
            "blue_group": list(self.blue_group),
            "blue_group_reason": self.blue_group_reason,
            "purple_group": list(self.purple_group),
            "purple_group_reason": self.purple_group_reason,
        }


def validate_solution(solution, puzzle_words):
    '''
    Validate the solution

    If the solutions are not a subset of the puzzle words, raise an error
    '''
    # check that each group has 4 words

    if len(solution.yellow_group) != 4:
        raise ValidationError("Yellow group must have 4 words.")
    if len(solution.green_group) != 4:
        raise ValidationError("Green group must have 4 words.")
    if len(solution.blue_group) != 4:
        raise ValidationError("Blue group must have 4 words.")
    if len(solution.purple_group) != 4:
        raise ValidationError("Purple group must have 4 words.")

    puzzle_words_set = set(puzzle_words)
    puzzle_words_set -= set(solution.yellow_group)
    puzzle_words_set -= set(solution.green_group)
    puzzle_words_set -= set(solution.blue_group)
    puzzle_words_set -= set(solution.purple_group)
    if len(puzzle_words_set) != 0:
        raise ValidationError(
            "Solution groups must be a subset of the puzzle words."
        )
    return solution


# Nodes
@task(retry=retry_policy)
def llm_call_generator(solver, puzzle_words: list[str], feedback: Feedback):
    """LLM generates a connections guess"""

    puzzle_text = ""
    for i in range(4):
        puzzle_text += ', '.join(puzzle_words[i * 4 : (i + 1) * 4]) + '\n'

    human_message = "Connections: \n" + puzzle_text

    if feedback:
        human_message += f"\nFeedback: {feedback.advice}"

    try:
        solution = solver.invoke(human_message)
    except Exception as e:
        print(e)
        raise e
    print(solution)
    validate_solution(solution, puzzle_words)
    return solution


@task(retry=retry_policy)
def llm_call_evaluator(evaluator, solution: Solution):
    """LLM evaluates the picks"""

    human_message = (
        f"Solutions:\n"
        f"Yellow Group Reason: {solution.yellow_group_reason}\n"
        f"Yellow Group: {', '.join(solution.yellow_group)}\n"
        f"Green Group Reason: {solution.green_group_reason}\n"
        f"Green Group: {', '.join(solution.green_group)}\n"
        f"Blue Group Reason: {solution.blue_group_reason}\n"
        f"Blue Group: {', '.join(solution.blue_group)}\n"
        f"Purple Group Reason: {solution.purple_group_reason}\n"
        f"Purple Group: {', '.join(solution.purple_group)}\n"
    )

    feedback = evaluator.invoke(human_message)
    return feedback


@entrypoint()
def optimizer_workflow(input):
    solver = input["solver"]
    evaluator = input["evaluator"]
    puzzle_words = input["puzzle_words"]

    tries = 0
    solution = []
    feedback = None
    submit = False
    while submit is False and tries < MAX_TRIES:
        tries += 1
        print("getting solution")
        solution = llm_call_generator(solver, puzzle_words, feedback).result()
        if tries < MAX_TRIES:
            # no need to evaluate the last solution
            print("evaluating solution")
            feedback = llm_call_evaluator(evaluator, solution).result()
            print(feedback)
            submit = feedback.submit

    if tries >= MAX_TRIES:
        print("Last solution submitted by default")
    else:
        print("Solution submitted")
    return solution


def solve_puzzle(solver, evaluator, puzzle_words, solution):
    print(puzzle_words)
    error = None

    # Invoke
    input = {
        "solver": solver,
        "evaluator": evaluator,
        "puzzle_words": puzzle_words,
    }
    try:
        for step in optimizer_workflow.stream(input, stream_mode="updates"):
            print(step)
            print("\n")
    except Exception as e:
        step = {"optimizer_workflow": None}
        print(e)
        error = str(e)

    if step['optimizer_workflow'] is None:
        print("Failed to find a solution")
        if not error:
            error = "Failed to find a solution"
    else:
        solution = step['optimizer_workflow']
        # solution_set = {frozenset(group) for group in solution}
    return solution, error


def main():
    load_dotenv()

    if PROMPT_SETTINGS['model'].startswith('claude'):
        llm = ChatAnthropic(model=PROMPT_SETTINGS['model'])
    else:
        llm = ChatOpenAI(
            model=PROMPT_SETTINGS['model'],
            temperature=PROMPT_SETTINGS['temperature'],
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_SETTINGS['prompts']['generator']['system']),
            ("human", "{input}"),
        ]
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_SETTINGS['prompts']['evaluator']['system']),
            ("human", "{input}"),
        ]
    )

    solver_structured = llm.with_structured_output(Solution)
    evaluator_structured = llm.with_structured_output(Feedback)
    solver = prompt | solver_structured
    evaluator = prompt | evaluator_structured

    connections = load_json(CONNECTIONS_FILEPATH)

    stats = {}
    if os.path.exists(STATS_FILEPATH):
        stats = load_json(STATS_FILEPATH)

    NUMBER_OF_PUZZLES = 40
    for i in range(NUMBER_OF_PUZZLES):
        if str(i) in stats:
            print("skipping", i)
            continue
        print("solving", i)
        solution = set(
            [
                frozenset(solution['words'])
                for solution in connections[i]['solution']['groups']
            ]
        )
        submitted_solution, error = solve_puzzle(
            solver,
            evaluator,
            connections[i]['words'],
            solution,
        )

        if error or not isinstance(submitted_solution, Solution):
            result = "error"
            solution = None
        else:
            result = (
                "correct"
                if submitted_solution.comparable_solution == solution
                else "incorrect"
            )
            solution = submitted_solution.serializable_solution

        stats[i] = {
            "result": result,
            "error": error,
            "submitted_solution": solution,
        }

        total = len(stats)
        correct = sum(
            1 for stat in stats.values() if stat["result"] == "correct"
        )
        incorrect = sum(
            1 for stat in stats.values() if stat["result"] == "incorrect"
        )
        failed = sum(1 for stat in stats.values() if stat["result"] == "error")

        print("percent correct", 100 * correct / total, '\n')
        print("errors: ", 100 * failed / total)
        print("incorrect: ", 100 * incorrect / total)
        print("\n")

        dump_json(stats, STATS_FILEPATH)


if __name__ == "__main__":
    main()
