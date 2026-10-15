# OnlyConnectBench

An [Inspect AI](https://inspect.aisi.org.uk/) evaluation built on the "connecting wall" round of the BBC quiz show [Only Connect](https://en.wikipedia.org/wiki/Only_Connect). A wall is 16 words or phrases that split into 4 hidden groups of 4. The model is shown the 16 items in shuffled order and asked to name the 4 connections.

The wall data come from the [OCW dataset](https://github.com/TaatiTeam/OCW) (Alavi Naeini et al., NeurIPS 2023, [arXiv:2306.11167](https://arxiv.org/abs/2306.11167)), which collected 618 walls from seasons 1 to 15 together with the human contestants' results.

## Usage

```bash
uv sync
uv run inspect eval --model openai/gpt-5-nano
```

Inspect ["task args"](https://inspect.aisi.org.uk/options.html#tasks-and-solvers) can be used to control the following parameters:

- `seed` (defaults to `0`): seed for shuffling the 16 words within each wall
- `grader_model` (defaults to the model under test)

## Dataset

The data from the [OCW repository](https://github.com/TaatiTeam/OCW) is bundled in `src/only_connect_bench/data/OCW.json`. An example wall (season 1, episode 3):

> Northern, District, Circle, Jubilee, Western, English, Bareback, Jumping, Southern, Maltese, Victoria, Red, Eastern, Central, Mountain, Pacific

1. London Underground lines: Northern, District, Circle, Jubilee
2. Horse riding styles: Western, English, Bareback, Jumping
3. \_\_\_ cross: Southern, Maltese, Victoria, Red
4. USA time zones: Eastern, Central, Mountain, Pacific
