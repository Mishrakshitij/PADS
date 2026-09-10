#!/usr/bin/env python3
"""Verify released CSVs, aggregate counts, and source-file integrity.

Uses only the Python standard library. The optional original ZIP is never
extracted; it also verifies source hashes and original field preservation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from contextlib import nullcontext
from pathlib import Path


SOURCE_HEADER = ["", "id", "prediction", "pred_score", "conversation_id", "root", "speaker", "text", "turnNumber"]
ANNOTATION_HEADER = SOURCE_HEADER + ["politeness_label", "politeness_label_name", "annotation_status"]
LABELS = {"0": "impolite", "1": "somewhat_impolite", "2": "somewhat_polite", "3": "polite"}
SUMMARY_HEADER = ["domain", "source_rows", "annotated_rows", "unscorable_rows"] + [f"politeness_label_{i}_rows" for i in range(4)]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_stream(handle)


def sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def verify(dataset_dir: Path, source_archive: Path | None = None) -> None:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] == 3:
        verify_annotations(dataset_dir, manifest, source_archive)
        return
    check(manifest["format_version"] == 1, "Unsupported manifest format")
    check(manifest["encoding"] == "utf-8", "Unsupported CSV encoding")
    expected_header = manifest["header"]
    with (dataset_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    summary = {row["domain"]: row for row in summary_rows}
    domains = [item["domain"] for item in manifest["domains"]]
    check(len(domains) == len(set(domains)), "Duplicate manifest domains")
    check(len(summary) == len(summary_rows), "Duplicate summary domains")
    check(set(summary) == set(domains), "Summary/manifest domains differ")
    csv_paths = set()
    total_rows = 0

    for item in manifest["domains"]:
        domain = item["domain"]
        source_hash = hashlib.sha256()
        source_bytes = 0
        rows = 0
        header_bytes = None
        conversations = set()
        ids = set()
        predictions = Counter()
        empty_speaker_rows = 0
        empty_text_rows = 0
        noninteger_turn_rows = 0
        text_characters = 0
        check(bool(item["files"]), f"{domain}: no files")

        for info in item["files"]:
            name = info["path"]
            check(Path(name).name == name, f"Invalid dataset filename: {name}")
            check(name not in csv_paths, f"Duplicate dataset filename: {name}")
            csv_paths.add(name)
            path = dataset_dir / name
            check(path.stat().st_size == info["bytes"], f"{name}: size differs")
            check(info["bytes"] < 50 * 1024 * 1024, f"{name}: exceeds 50 MiB")
            file_hash = hashlib.sha256()
            file_rows = 0
            with path.open("rb") as handle:
                raw_header = handle.readline()
                file_hash.update(raw_header)
                header = next(csv.reader([raw_header.decode("utf-8")], strict=True))
                check(header == expected_header, f"{name}: header differs")
                if header_bytes is None:
                    header_bytes = raw_header
                    source_hash.update(raw_header)
                    source_bytes += len(raw_header)
                else:
                    check(raw_header == header_bytes, f"{name}: header bytes differ")

                def lines():
                    nonlocal source_bytes
                    for raw in handle:
                        file_hash.update(raw)
                        source_hash.update(raw)
                        source_bytes += len(raw)
                        yield raw.decode("utf-8")

                for record in csv.reader(lines(), strict=True):
                    file_rows += 1
                    check(len(record) == len(expected_header), f"{name}: row {file_rows} has wrong width")
                    row = dict(zip(expected_header, record))
                    ids.add(row["id"])
                    conversations.add(row["conversation_id"])
                    predictions[row["prediction"]] += 1
                    empty_speaker_rows += row["speaker"] == ""
                    empty_text_rows += row["text"] == ""
                    text_characters += len(row["text"])
                    try:
                        int(row["turnNumber"])
                    except ValueError:
                        noninteger_turn_rows += 1
            check(file_hash.hexdigest() == info["sha256"], f"{name}: SHA-256 differs")
            check(file_rows == info["rows"], f"{name}: row count differs")
            rows += file_rows

        check(rows == item["source"]["rows"], f"{domain}: total row count differs")
        check(source_bytes == item["source"]["bytes"], f"{domain}: reconstructed source size differs")
        check(source_hash.hexdigest() == item["source"]["sha256"], f"{domain}: reconstructed source SHA-256 differs")
        expected_summary = {
            "rows": rows,
            "conversations": len(conversations),
            "unique_ids": len(ids),
            "prediction_0_rows": predictions["0"],
            "prediction_1_rows": predictions["1"],
            "prediction_2_rows": predictions["2"],
            "prediction_3_rows": predictions["3"],
            "empty_speaker_rows": empty_speaker_rows,
            "empty_text_rows": empty_text_rows,
            "noninteger_turn_number_rows": noninteger_turn_rows,
            "source_bytes": source_bytes,
            "csv_files": len(item["files"]),
        }
        for key, value in expected_summary.items():
            check(int(summary[domain][key]) == value, f"{domain}: summary {key} differs")
        check(float(summary[domain]["mean_text_characters"]) == round(text_characters / rows, 6), f"{domain}: mean text length differs")
        check(dict(predictions) == item["statistics"]["prediction_counts"], f"{domain}: label counts differ")
        total_rows += rows
        print(f"{domain}: {rows:,} rows in {len(item['files'])} file(s); source bytes verified")

    actual_paths = {path.name for path in dataset_dir.glob("*.csv") if path.name != "summary.csv"}
    check(actual_paths == csv_paths, "CSV files differ from manifest inventory")

    if source_archive is not None:
        archive_info = manifest["source_archive"]
        check(source_archive.stat().st_size == archive_info["bytes"], "Source ZIP size differs")
        check(sha256_file(source_archive) == archive_info["sha256"], "Source ZIP SHA-256 differs")
        with zipfile.ZipFile(source_archive) as archive:
            for item in manifest["domains"]:
                source = item["source"]
                with archive.open(source["archive_member"]) as handle:
                    check(sha256_stream(handle) == source["sha256"], f"{item['domain']}: original ZIP member SHA-256 differs")
        print("Original ZIP and all six source members verified")

    print(f"Verified {len(domains)} domains, {len(csv_paths)} CSV files, and {total_rows:,} data rows")


def nonnegative_integer(value, name: str) -> int:
    check(type(value) is int and value >= 0, f"{name}: expected a nonnegative integer")
    return value


def digest_value(value, name: str) -> str:
    check(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
          f"{name}: invalid SHA-256")
    return value


def local_file(dataset_dir: Path, name: str) -> Path:
    check(isinstance(name, str) and name not in ("", ".", "..") and Path(name).name == name,
          f"Invalid dataset filename: {name}")
    path = dataset_dir / name
    check(path.resolve().parent == dataset_dir.resolve(), f"Dataset path escapes directory: {name}")
    return path


def verify_release_metadata(dataset_dir: Path, manifest: dict) -> dict:
    """Check the release schema, label mapping, and saved source inventory."""
    required = {"encoding", "header", "created_at", "annotation_status", "labels",
                "original_prediction_column", "unscorable_input_policy", "source_manifest",
                "source_manifest_file", "source_manifest_sha256", "domains", "max_shard_bytes",
                "summary_sha256"}
    check(required.issubset(manifest),
          f"Missing release metadata: {', '.join(sorted(required.difference(manifest)))}")
    check(manifest["encoding"] == "utf-8", "Unsupported CSV encoding")
    check(manifest["header"] == ANNOTATION_HEADER, "Release manifest header differs")
    check(manifest["annotation_status"] in ("complete", "partial"), "Invalid release status")
    check(manifest["labels"] == LABELS, "Release label mapping differs")
    for field in ("created_at", "original_prediction_column", "unscorable_input_policy"):
        check(isinstance(manifest[field], str) and bool(manifest[field].strip()),
              f"Invalid release metadata: {field}")
    source_hash = digest_value(manifest["source_manifest_sha256"], "source_manifest_sha256")
    source = manifest["source_manifest"]
    check(isinstance(source, dict), "Invalid source manifest")
    check(source["format_version"] == 1 and source["encoding"] == "utf-8", "Invalid source manifest format")
    check(source["header"] == SOURCE_HEADER, "Original source header differs")
    source_file = local_file(dataset_dir, manifest["source_manifest_file"])
    check(sha256_file(source_file) == source_hash, "Saved source manifest SHA-256 differs")
    check(json.loads(source_file.read_text(encoding="utf-8")) == source,
          "Saved and embedded source manifests differ")
    nonnegative_integer(source["source_archive"]["bytes"], "source_archive.bytes")
    digest_value(source["source_archive"]["sha256"], "source_archive.sha256")
    return source


def original_domain_rows(archive: zipfile.ZipFile, item: dict):
    """Stream original CSV records and verify raw member bytes upon exhaustion."""
    source, domain = item["source"], item["domain"]
    digest, size, count = hashlib.sha256(), 0, 0
    with archive.open(source["archive_member"]) as handle:
        def lines():
            nonlocal size
            for raw in handle:
                digest.update(raw)
                size += len(raw)
                yield raw.decode("utf-8")

        reader = csv.reader(lines(), strict=True)
        check(next(reader) == SOURCE_HEADER, f"{domain}: original ZIP member header differs")
        for count, row in enumerate(reader, 1):
            check(len(row) == len(SOURCE_HEADER), f"{domain}: original ZIP row {count} has wrong width")
            yield row
    check(count == source["rows"], f"{domain}: original ZIP member row count differs")
    check(size == source["bytes"], f"{domain}: original ZIP member size differs")
    check(digest.hexdigest() == source["sha256"], f"{domain}: original ZIP member SHA-256 differs")


def verify_annotations(dataset_dir: Path, manifest: dict, source_archive: Path | None = None) -> None:
    """Verify a complete annotation extension or an explicitly partial export."""
    source_manifest = verify_release_metadata(dataset_dir, manifest)
    source_domains = {item["domain"]: item for item in source_manifest["domains"]}
    domains = [item["domain"] for item in manifest["domains"]]
    check(bool(domains) and len(domains) == len(set(domains)), "Empty or duplicate annotation domains")
    check(len(source_domains) == len(source_manifest["domains"]), "Duplicate source manifest domains")
    check(set(domains) == set(source_domains), "Annotation/source manifest domains differ")
    max_bytes = nonnegative_integer(manifest["max_shard_bytes"], "max_shard_bytes")
    check(0 < max_bytes <= 48 * 1024 * 1024, "Invalid annotation shard byte limit")

    summary_path = dataset_dir / "summary.csv"
    check(sha256_file(summary_path) == digest_value(manifest["summary_sha256"], "summary_sha256"),
          "Summary SHA-256 differs")
    with summary_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        check(next(reader) == SUMMARY_HEADER, "Annotation summary header differs")
        summary = {}
        for row in reader:
            check(len(row) == len(SUMMARY_HEADER), "Annotation summary row has wrong width")
            check(row[0] not in summary, "Duplicate summary domains")
            summary[row[0]] = {key: int(value) for key, value in zip(SUMMARY_HEADER[1:], row[1:])}
    check(set(summary) == set(domains), "Summary/manifest domains differ")
    if source_archive is not None:
        info = source_manifest["source_archive"]
        check(source_archive.stat().st_size == info["bytes"], "Source ZIP size differs")
        check(sha256_file(source_archive) == info["sha256"], "Source ZIP SHA-256 differs")

    total_rows = total_source = total_unscorable = 0
    total_labels = Counter({label: 0 for label in LABELS})
    summary_labels = Counter({label: 0 for label in LABELS})
    csv_paths = set()
    complete = manifest["annotation_status"] == "complete"
    with zipfile.ZipFile(source_archive) if source_archive is not None else nullcontext(None) as archive:
        for item in manifest["domains"]:
            domain = item["domain"]
            original = source_domains[domain]
            check(isinstance(original["source"]["archive_member"], str)
                  and bool(original["source"]["archive_member"]), f"{domain}: missing original archive member")
            nonnegative_integer(original["source"]["bytes"], f"{domain}.source.bytes")
            digest_value(original["source"]["sha256"], f"{domain}.source.sha256")
            expected_source = nonnegative_integer(original["source"]["rows"], f"{domain}.source.rows")
            check(sum(nonnegative_integer(info["rows"], f"{domain}.source.file.rows") for info in original["files"])
                  == expected_source, f"{domain}: source manifest shard counts differ")
            check(nonnegative_integer(item["source_rows"], f"{domain}.source_rows") == expected_source,
                  f"{domain}: source row count differs")
            source_rows = original_domain_rows(archive, original) if archive is not None else None
            counts, rows, unscorable = Counter({label: 0 for label in LABELS}), 0, 0
            for info in item["files"]:
                name = info["path"]
                path = local_file(dataset_dir, name)
                check(name.endswith(".csv") and name != "summary.csv", f"Invalid annotation CSV filename: {name}")
                check(name not in csv_paths, f"Duplicate dataset filename: {name}")
                csv_paths.add(name)
                size = nonnegative_integer(info["bytes"], f"{name}.bytes")
                check(path.stat().st_size == size, f"{name}: size differs")
                check(size <= max_bytes, f"{name}: exceeds shard byte limit")
                check(sha256_file(path) == digest_value(info["sha256"], f"{name}.sha256"), f"{name}: SHA-256 differs")
                file_rows = 0
                with path.open(encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle, strict=True)
                    check(next(reader) == ANNOTATION_HEADER, f"{name}: header differs")
                    for file_rows, record in enumerate(reader, 1):
                        check(len(record) == len(ANNOTATION_HEADER), f"{name}: row {file_rows} has wrong width")
                        label, label_name, status = record[-3:]
                        scorable = any(char.isprintable() and not char.isspace() for char in record[7])
                        if scorable:
                            check(status == "annotated" and label in LABELS and label_name == LABELS.get(label),
                                  f"{name}: row {file_rows} has invalid label/name/status")
                            counts[label] += 1
                        else:
                            check((label, label_name, status) == ("", "", "unscorable_empty_text"),
                                  f"{name}: row {file_rows} unscorable text must have missing labels")
                            unscorable += 1
                        if source_rows is not None:
                            if complete:
                                check(next(source_rows, None) == record[:len(SOURCE_HEADER)],
                                      f"{domain}: source fields or row order differ at exported row {rows + file_rows}")
                            else:
                                for source_record in source_rows:
                                    if source_record == record[:len(SOURCE_HEADER)]:
                                        break
                                else:
                                    raise ValueError(f"{domain}: exported row {rows + file_rows} is not an ordered source record")
                check(file_rows == nonnegative_integer(info["rows"], f"{name}.rows"), f"{name}: row count differs")
                check(file_rows > 0, f"{name}: empty annotation shard")
                rows += file_rows
            if source_rows is not None:
                for _ in source_rows:
                    check(not complete, f"{domain}: complete export omits original source rows")
            check(rows <= expected_source, f"{domain}: export exceeds source row count")
            if complete:
                check(rows == expected_source, f"{domain}: complete export omits source rows")
            check(rows == nonnegative_integer(item["exported_rows"], f"{domain}.exported_rows"), f"{domain}: exported row count differs")
            check(sum(counts.values()) == nonnegative_integer(item["annotated_rows"], f"{domain}.annotated_rows"),
                  f"{domain}: annotated row count differs")
            check(unscorable == nonnegative_integer(item["unscorable_rows"], f"{domain}.unscorable_rows"),
                  f"{domain}: unscorable row count differs")
            expected_counts = item["statistics"]["politeness_label_counts"]
            check(set(expected_counts) == set(LABELS), f"{domain}: label count keys differ")
            for label in LABELS:
                nonnegative_integer(expected_counts[label], f"{domain}.label_count.{label}")
            check(dict(counts) == expected_counts, f"{domain}: label counts differ")
            expected_summary = {"source_rows": expected_source, "annotated_rows": sum(counts.values()), "unscorable_rows": unscorable,
                                **{f"politeness_label_{label}_rows": counts[label] for label in LABELS}}
            check(summary[domain] == expected_summary, f"{domain}: summary counts differ")
            total_rows += rows
            total_source += expected_source
            total_unscorable += unscorable
            total_labels.update(counts)
            summary_labels.update({label: summary[domain][f"politeness_label_{label}_rows"] for label in LABELS})
            print(f"{domain}: {rows:,}/{expected_source:,} source rows; {sum(counts.values()):,} labels, {unscorable:,} unscorable")
    check(total_labels == summary_labels, "Aggregate summary label counts differ")
    check(total_rows == sum(total_labels.values()) + total_unscorable, "Aggregate annotation coverage differs")
    check(complete or total_rows < total_source, "Partial release contains all source rows")
    actual_paths = {path.name for path in dataset_dir.glob("*.csv") if path.name != "summary.csv"}
    check(actual_paths == csv_paths, "CSV files differ from manifest inventory")
    if source_archive is not None:
        print("Original ZIP, source member hashes, and every exported source field verified")
    print(f"Verified {manifest['annotation_status']} annotation release: {len(domains)} domains, {len(csv_paths)} CSV files, "
          f"{total_rows:,} data rows; label counts {dict(total_labels)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parents[1] / "datasets")
    parser.add_argument("--source-archive", type=Path, help="Optionally verify against RL_PADS_Data.zip")
    args = parser.parse_args()
    try:
        verify(args.dataset_dir, args.source_archive)
    except (OSError, ValueError, KeyError, TypeError, StopIteration, csv.Error, zipfile.BadZipFile) as error:
        print(f"Dataset verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
