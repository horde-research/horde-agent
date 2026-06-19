from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.data.image_dedup import deduplicate_image_records


def _write_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (16, 16)) -> None:
    Image.new("RGB", size, color=color).save(path)


def test_image_dedup_clusters_exact_and_embedding_duplicates(tmp_path: Path) -> None:
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_c = tmp_path / "c.jpg"
    image_d = tmp_path / "d.jpg"
    _write_image(image_a, (255, 0, 0), size=(16, 16))
    image_b.write_bytes(image_a.read_bytes())
    _write_image(image_c, (0, 255, 0), size=(32, 32))
    _write_image(image_d, (0, 0, 255), size=(24, 24))

    records = [
        {"file_path": str(image_a), "width": "16", "height": "16"},
        {"file_path": str(image_b), "width": "16", "height": "16"},
        {"file_path": str(image_c), "width": "32", "height": "32"},
        {"file_path": str(image_d), "width": "24", "height": "24"},
    ]

    def fake_embeddings(paths: list[Path]) -> list[list[float]]:
        assert paths == [image_a, image_b, image_c, image_d]
        return [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.99, 0.01],
        ]

    result = deduplicate_image_records(
        records,
        output_dir=tmp_path,
        threshold=0.95,
        embedding_fn=fake_embeddings,
    )

    kept_paths = {record["file_path"] for record in result["records"]}
    assert kept_paths == {str(image_a), str(image_c)}
    report = result["report"]
    assert report["num_input_records"] == 4
    assert report["num_kept_records"] == 2
    assert report["num_removed_records"] == 2
    assert report["exact_duplicate_pair_count"] == 1
    assert report["embedding_duplicate_pair_count"] == 1
    assert report["num_duplicate_clusters"] == 2
