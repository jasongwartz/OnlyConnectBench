import json

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from only_connect_bench import only_connect_bench
from only_connect_bench.only_connect_bench import (
    DATA_FILE,
    load_dataset,
    record_to_sample,
)


def test_words_are_shuffled_deterministically_per_seed() -> None:
    record = json.loads(DATA_FILE.read_text(encoding="utf-8"))["dataset"][0]
    sample = record_to_sample(record, seed=42)

    assert sample.input == record_to_sample(record, seed=42).input
    assert sample.input != record_to_sample(record, seed=7).input

    assert sample.metadata is not None
    words = sample.metadata["words"]
    assert words != record["words"]
    assert sorted(words) == sorted(record["words"])
    for word in record["words"]:
        assert word in str(sample.input)


def test_bundled_dataset_is_complete() -> None:
    dataset = load_dataset()

    assert len(dataset) == 618
    assert len({sample.id for sample in dataset}) == 618
    for sample in dataset:
        assert sample.metadata is not None
        assert len(sample.metadata["words"]) == 16
        assert str(sample.target).count("\n") == 3


def test_end_to_end_with_mock_models() -> None:
    answer = ModelOutput.from_content(
        model="mockllm/model",
        content="Made from rubber\nPrison slang\nCalled Nick\nQuarks",
    )
    grade = ModelOutput.from_content(model="mockllm/model", content="GRADE: C")

    logs = eval(
        only_connect_bench(
            grader_model=get_model("mockllm/model", custom_outputs=[grade])
        ),
        model=get_model("mockllm/model", custom_outputs=[answer]),
        limit=1,
        display="none",
    )

    assert len(logs) == 1
    assert logs[0].status == "success"
    assert logs[0].results is not None
    assert logs[0].results.scores[0].metrics["accuracy"].value == 1.0
