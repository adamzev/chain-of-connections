import os
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.func import RetryPolicy, entrypoint, task
from pydantic import BaseModel, Field

from src.utils.json_helpers import dump_json, load_json

CONNECTIONS_FILEPATH = "data/connections.jsonl"
STATS_FILEPATH = "output/stats.json"
MAX_TRIES = 3


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
        description="Forward looking advice regarding how to solve this puzzle. The advice should not reference the present solution groups as you are advising a new solver.",
    )


class Solution(BaseModel):
    yellow_group: set[str] = Field(
        description="The 4 words that go best together for the yellow group.",
        min_items=4,
        max_items=4,
    )
    yellow_group_reason: str = Field(
        description="The very brief reason why the words go together for the yellow group.",
    )
    green_group: set[str] = Field(
        description="The 4 words that go best together for the green group.",
        min_items=4,
        max_items=4,
    )
    green_group_reason: str = Field(
        description="The very brief reason why the words go together for the green group.",
    )
    blue_group: set[str] = Field(
        description="The 4 words that go best together for the blue group.",
        min_items=4,
        max_items=4,
    )
    blue_group_reason: str = Field(
        description="The very brief reason why the words go together for the blue group.",
    )
    purple_group: set[str] = Field(
        description="The 4 words that go best together for the purple group.",
        min_items=4,
        max_items=4,
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
    Pydantic will handle the rest of the validation regarding 4 words in each group
    '''
    puzzle_words_set = set(puzzle_words)
    puzzle_words_set -= solution.yellow_group
    puzzle_words_set -= solution.green_group
    puzzle_words_set -= solution.blue_group
    puzzle_words_set -= solution.purple_group
    if len(puzzle_words_set) != 0:
        raise ValidationError(
            "Solution groups must be a subset of the puzzle words."
        )
    return solution


# Nodes
@task(retry=retry_policy)
def llm_call_generator(solver, puzzle_words: list[str], feedback: Feedback):
    """LLM generates a connections guess"""

    prompt = (
        "You are trying to solve a Connections word puzzle. To solve a connections word puzzle, you split 16 words into 4 groups of 4 based on which words share a connection. "
        "Some connections are more obvious than others. However, once you see the connection, it should be clear. \n"
        "The words may include proper nouns. \n"
        "Each group is assigned a color name which indicates how straightforward the connection is. The groups are labeled rated yellow, green, blue, or purple, with yellow being the most straightforward and purple being the most difficult. \n"
        "Common connections types are the yellow group having a shared meaning and the green group having a shared category. The blue group may be components of a shared category."
        "The purple group usually requires putting the word into a phrase or modifying the words (pet types with a letter removed for example)"
        "However, these are loose guidelines and other types of connections exist such as appearing in the same piece of popular culture or wordplay (homonyms).\n"
        "Out of all the following options, pick 4 groups of 4 words that form a connection together and state the connection.  \n"
        f"Words:\n {"\n".join(puzzle_words)}"
    )

    if feedback:
        f"\nFeedback: {feedback.advice}"

    try:
        solution = solver.invoke(prompt)
    except Exception as e:
        print(e)
        raise e
    solution = solver.invoke(prompt)
    print(solution)
    validate_solution(solution, puzzle_words)
    return solution


@task(retry=retry_policy)
def llm_call_evaluator(evaluator, solution: Solution):
    """LLM evaluates the picks"""
    feedback = evaluator.invoke(
        "You are trying to evaulate a solution to a Connections word puzzle. To solve a connections word puzzle, you split 16 words into 4 groups of 4 based on which words share a connection. "
        "Some connections are more obvious than others. However, once you see the connection, it should be clear. \n"
        "The words may include proper nouns. \n"
        "Each group is assigned a color name which indicates how straightforward the connection is. The groups are labeled rated yellow, green, blue, or purple, with yellow being the most straightforward and purple being the most difficult. \n"
        "Common connections types are the yellow group having a shared meaning and the green group having a shared category. The blue group may be components of a shared category."
        "The purple group usually requires putting the word into a phrase or modifying the words (pet types with a letter removed for example)"
        "However, these are loose guidelines and other types of connections exist such as appearing in the same piece of popular culture or wordplay (homonyms).\n"
        f"Evaluate if the connections are correct. The solution is a valid solution but may have items that have been put in the wrong group.\n"
        f"For example, if the reason a group of words is connected is complicated or involves the word 'and' (for example 'foods **and** stuffy people smell')  it is likely incorrect."
        f"If the solution is correct, submit the results (submit=True) and skip providing advice. If the solution is incorrect, provide specific advice to a new solver regarding what words may go together and which words do not. \n"
        f"If you reference a group, specify what words are in the group as the new solver cannot see the previous solution. \n"
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
    return feedback


@entrypoint()
def optimizer_workflow(input):
    solver = input["solver"]
    evaluator = input["evaluator"]
    puzzle_words = input["puzzle_words"]
    starting_puzzle_words = puzzle_words.copy()
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
    # Augment the LLM with schema for structured output
    # claude-3-5-haiku-latest
    # claude-3-5-sonnet-latest
    # llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    solver = llm.with_structured_output(Solution)
    evaluator = llm.with_structured_output(Feedback)
    stats = {}
    connections = load_json(CONNECTIONS_FILEPATH)
    stats = load_json(STATS_FILEPATH)
    for i in range(40):
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
