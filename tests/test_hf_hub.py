from __future__ import annotations

from datasets import Dataset, DatasetDict

from core.hf_hub import push_dataset


def test_push_dataset_preserves_datasetdict_splits(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    dataset_path = tmp_path / "dataset"
    DatasetDict(
        {
            "train": Dataset.from_list([{"text": "train"}]),
            "validation": Dataset.from_list([{"text": "validation"}]),
        }
    ).save_to_disk(str(dataset_path))
    calls: list[dict[str, object]] = []

    def fake_push_to_hub(self, repo_id, *, private=True, token=None):
        calls.append(
            {
                "repo_id": repo_id,
                "private": private,
                "token": token,
                "splits": sorted(self.keys()),
            }
        )

    monkeypatch.setattr(DatasetDict, "push_to_hub", fake_push_to_hub)

    repo_id = push_dataset(str(dataset_path), "test-dataset", username="test-owner")

    assert repo_id == "test-owner/test-dataset"
    assert calls == [
        {
            "repo_id": "test-owner/test-dataset",
            "private": True,
            "token": None,
            "splits": ["train", "validation"],
        }
    ]
