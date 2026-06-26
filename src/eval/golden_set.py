import json
from pathlib import Path
from dataclasses import dataclass

GOLDEN_SET_PATH = Path("src/eval/golden_set.json")


@dataclass
class EvalQuestion:
    id: int
    question: str
    ground_truth: str
    difficulty: str
    source_files: list[str]


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[EvalQuestion]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return [
        EvalQuestion(
            id=item["id"],
            question=item["question"],
            ground_truth=item["ground_truth"],
            difficulty=item["difficulty"],
            source_files=item["source_files"],
        )
        for item in data
    ]


def filter_by_difficulty(
    questions: list[EvalQuestion], difficulty: str
) -> list[EvalQuestion]:
    return [q for q in questions if q.difficulty == difficulty]


if __name__ == "__main__":
    questions = load_golden_set()
    print(f"Total questions: {len(questions)}\n")

    for diff in ["easy", "medium", "hard", "out_of_scope"]:
        subset = filter_by_difficulty(questions, diff)
        print(f"  {diff}: {len(subset)}")

    print("\nSample questions:")
    for q in questions[:3]:
        print(f"  [{q.difficulty}] {q.question}")
