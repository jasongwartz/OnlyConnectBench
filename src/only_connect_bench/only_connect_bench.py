import json
import random
from importlib.resources import files
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import Model
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate

DATA_FILE = files("only_connect_bench").joinpath("data", "OCW.json")

TASK_PROMPT = """The following is a puzzle from a quiz.

You will be given 16 words or phrases, separated by commas. They form 4 groups of 4 words or phrases, where each group shares a connection.
Your task is to identify the 4 connections.

For example, if four of the words or phrases were "The Louvre", "The Seine", "Musée d'Orsay" and "Arc de Triomphe", one of your answers should be "things in Paris".

Reply with only the names or descriptions of the four connections, one per line.

The 16 words or phrases are:
{words}"""

GRADER_TEMPLATE = """You are marking an answer to a "connecting wall" puzzle from the quiz show Only Connect.

The puzzle presented 16 words or phrases that form 4 groups of 4 related items. The contestant was asked to name the connection for each group.

The correct connections, one per line, are:
{criterion}

The contestant's answer was:
{answer}

Count how many of the 4 correct connections the contestant identified. A connection counts as identified if the contestant described the same idea, even in different words or at a different level of detail.

Mark the answer as correct if all 4 connections were identified, partially correct if 1 to 3 were identified, and incorrect if none were.

{instructions}"""


def record_to_sample(record: dict[str, Any], seed: int = 0) -> Sample:
    """Turn one OCW wall into a Sample.

    The dataset lists the 16 words grouped in order, so they are shuffled
    before being shown to the model. The shuffle is seeded from `seed` and the
    wall id so a given seed always produces the same prompt for a wall.
    """
    words = [str(word) for word in record["words"]]
    rng = random.Random(f"{seed}:{record['wall_id']}")
    rng.shuffle(words)

    connections = [str(connection) for connection in record["gt_connections"]]

    return Sample(
        id=record["wall_id"],
        input=TASK_PROMPT.format(words=", ".join(words)),
        target="\n".join(connections),
        metadata={
            "season": record["season"],
            "episode": record["episode"],
            "words": words,
            "groups": record["groups"],
            "human_performance": record.get("overall_human_performance"),
        },
    )


def load_dataset(seed: int = 0) -> MemoryDataset:
    """Load all 618 walls from the bundled OCW json."""
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    samples = [record_to_sample(record, seed=seed) for record in data["dataset"]]
    return MemoryDataset(samples=samples, name="OCW")


@task
def only_connect_bench(
    seed: int = 0,
    grader_model: str | Model | None = None,
) -> Task:
    """Name the four connections in an Only Connect wall.

    Args:
        seed: Seed for shuffling the 16 words within each wall.
        grader_model: Model used to mark answers. Defaults to the model being
            evaluated.
    """
    return Task(
        dataset=load_dataset(seed=seed),
        solver=generate(),
        scorer=model_graded_qa(
            template=GRADER_TEMPLATE,
            model=grader_model,
            partial_credit=True,
        ),
    )
