"""Offline checks of exported annotation integrity and original ZIP preservation."""
from collections import Counter
from contextlib import redirect_stdout
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = load_script("verify_dataset")
HEADER = verifier.SOURCE_HEADER


def csv_bytes(rows):
    handle = io.StringIO(newline="")
    csv.writer(handle, lineterminator="\r\n").writerows(rows)
    return handle.getvalue().encode("utf-8")


class DatasetReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.output = (self.root / name for name in ("source", "output"))
        self.source.mkdir()
        self.archive = self.root / "original.zip"
        texts = ["What is the code?", "Can you check?", "Thank you,\nthat helps!", "I appreciate your help, please continue.", " \t\x01\n", "What is the code?"]
        self.labels = {text: label for text, label in zip(texts, [0, 1, 2, 3, None, 0])}
        self.rows = [[str(i), "repeated-id", str(i % 2), "0.75", "conversation", "conversation", "customer", text, str(i)]
                     for i, text in enumerate(texts)]
        manifest = {"format_version": 1, "encoding": "utf-8", "header": HEADER, "domains": []}
        summaries = []
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for domain in ("airline", "media"):
                raw = csv_bytes([HEADER] + self.rows)
                member = f"original/{domain}.csv"
                archive.writestr(member, raw)
                groups = [self.rows] if domain == "airline" else [self.rows[:3], self.rows[3:]]
                files = []
                for part, rows in enumerate(groups, 1):
                    name = f"{domain}.part-{part}.csv"
                    path = self.source / name
                    path.write_bytes(csv_bytes([HEADER] + rows))
                    files.append({"path": name, "rows": len(rows), "bytes": path.stat().st_size, "sha256": verifier.sha256_file(path)})
                predictions = Counter(row[2] for row in self.rows)
                manifest["domains"].append({"domain": domain,
                    "source": {"archive_member": member, "rows": len(self.rows), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
                    "files": files, "statistics": {"prediction_counts": dict(predictions)}})
                summaries.append({"domain": domain, "rows": len(self.rows), "conversations": 1, "unique_ids": 1,
                    **{f"prediction_{i}_rows": predictions[str(i)] for i in range(4)},
                    "empty_speaker_rows": 0, "empty_text_rows": 0, "noninteger_turn_number_rows": 0,
                    "source_bytes": len(raw), "csv_files": len(files), "mean_text_characters": round(sum(map(len, texts)) / len(texts), 6)})
        manifest["source_archive"] = {"filename": self.archive.name, "bytes": self.archive.stat().st_size,
                                      "sha256": verifier.sha256_file(self.archive)}
        (self.source / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (self.source / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        self.write_release(self.output)

    def write_release(self, directory, keep_labels=None):
        """Build tiny releases with multiline fields and split source files."""
        directory.mkdir()
        source_raw = (self.source / "manifest.json").read_bytes()
        (directory / "source_manifest.json").write_bytes(source_raw)
        manifest = {"format_version": 3, "encoding": "utf-8", "header": verifier.ANNOTATION_HEADER,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "annotation_status": "complete" if keep_labels is None else "partial",
                    "labels": verifier.LABELS, "source_manifest_sha256": hashlib.sha256(source_raw).hexdigest(),
                    "original_prediction_column": "Retained unchanged.",
                    "unscorable_input_policy": "Empty labels for whitespace/control-only inputs.",
                    "source_manifest": json.loads(source_raw), "source_manifest_file": "source_manifest.json",
                    "domains": [], "max_shard_bytes": 512}
        summaries = [verifier.SUMMARY_HEADER]
        for domain in ("airline", "media"):
            exported, counts, files = [], Counter(), []
            for source_row in self.rows:
                label = self.labels[source_row[7]]
                if label is None:
                    exported.append(source_row + ["", "", "unscorable_empty_text"])
                elif keep_labels is None or label in keep_labels:
                    exported.append(source_row + [str(label), verifier.LABELS[str(label)], "annotated"])
                    counts[str(label)] += 1
            # Fixed pairs deliberately exercise source and output shard boundaries.
            for index in range(0, len(exported), 2):
                rows = exported[index:index + 2]
                name = f"{domain}_politeness.part-{index // 2 + 1:03d}.csv"
                path = directory / name
                path.write_bytes(csv_bytes([verifier.ANNOTATION_HEADER] + rows))
                files.append({"path": name, "rows": len(rows), "bytes": path.stat().st_size, "sha256": verifier.sha256_file(path)})
            manifest["domains"].append({"domain": domain, "source_rows": 6, "annotated_rows": sum(counts.values()),
                "unscorable_rows": 1, "exported_rows": len(exported), "files": files,
                "statistics": {"politeness_label_counts": {str(i): counts[str(i)] for i in range(4)}}})
            summaries.append([domain, 6, sum(counts.values()), 1] + [counts[str(i)] for i in range(4)])
        (directory / "summary.csv").write_bytes(csv_bytes(summaries))
        manifest["summary_sha256"] = verifier.sha256_file(directory / "summary.csv")
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def manifest(self):
        return json.loads((self.output / "manifest.json").read_text())

    def save_manifest(self, manifest):
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def verify(self, *, source=True, directory=None):
        with redirect_stdout(io.StringIO()):
            verifier.verify(directory or self.output, self.archive if source else None)

    def rewrite_first_record(self, transform):
        manifest = self.manifest()
        info = manifest["domains"][0]["files"][0]
        path = self.output / info["path"]
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        transform(rows[1])
        path.write_bytes(csv_bytes(rows))
        info.update(bytes=path.stat().st_size, sha256=verifier.sha256_file(path))
        self.save_manifest(manifest)

    def test_legacy_and_complete_release_verify_against_multiline_split_zip(self):
        self.verify(directory=self.source)
        self.verify()
        self.verify(source=False)
        manifest = self.manifest()
        self.assertEqual(sum(item["exported_rows"] for item in manifest["domains"]), 12)
        self.assertEqual(sum(item["unscorable_rows"] for item in manifest["domains"]), 2)
        self.assertGreater(len(manifest["domains"][1]["files"]), 1)

    def test_original_field_tampering_fails_even_if_export_hashes_are_updated(self):
        self.rewrite_first_record(lambda row: row.__setitem__(1, "different-id"))
        with self.assertRaisesRegex(ValueError, "source fields or row order differ"):
            self.verify()

    def test_source_row_order_is_checked_even_when_counts_and_hashes_match(self):
        manifest = self.manifest()
        info = manifest["domains"][0]["files"][0]
        path = self.output / info["path"]
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        rows[1], rows[2] = rows[2], rows[1]
        path.write_bytes(csv_bytes(rows))
        info.update(bytes=path.stat().st_size, sha256=verifier.sha256_file(path))
        self.save_manifest(manifest)
        self.verify(source=False)
        with self.assertRaisesRegex(ValueError, "source fields or row order differ"):
            self.verify()

    def test_label_name_and_status_must_match_the_numeric_label(self):
        for column, value in ((-3, "4"), (-2, "polite"), (-1, "pending")):
            with self.subTest(column=column):
                before = {path.name: path.read_bytes() for path in self.output.iterdir()}
                self.rewrite_first_record(lambda row: row.__setitem__(column, value))
                with self.assertRaisesRegex(ValueError, "invalid label/name/status"):
                    self.verify(source=False)
                for name, raw in before.items():
                    (self.output / name).write_bytes(raw)

    def test_unscorable_rows_cannot_be_replaced_by_label_zero(self):
        manifest = self.manifest()
        for info in manifest["domains"][0]["files"]:
            path = self.output / info["path"]
            with path.open(newline="") as handle:
                rows = list(csv.reader(handle))
            changed = False
            for row in rows[1:]:
                if row[-1] == "unscorable_empty_text":
                    row[-3:] = ["0", "impolite", "annotated"]
                    changed = True
            if changed:
                path.write_bytes(csv_bytes(rows))
                info.update(bytes=path.stat().st_size, sha256=verifier.sha256_file(path))
                break
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "unscorable text must have missing labels"):
            self.verify(source=False)

    def test_saved_source_manifest_tampering_and_malformed_hash_are_detected(self):
        manifest = self.manifest()
        path = self.output / manifest["source_manifest_file"]
        before = path.read_bytes()
        path.write_bytes(before + b" ")
        with self.assertRaisesRegex(ValueError, "source manifest SHA-256 differs"):
            self.verify(source=False)
        path.write_bytes(before)
        manifest["source_manifest_sha256"] = "not-a-sha256"
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "source_manifest_sha256: invalid SHA-256"):
            self.verify(source=False)

    def test_missing_saved_source_manifest_is_not_silently_ignored(self):
        manifest = self.manifest()
        (self.output / manifest["source_manifest_file"]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify(source=False)

    def test_saved_and_embedded_source_inventories_must_agree(self):
        manifest = self.manifest()
        manifest["source_manifest"]["source_archive"]["filename"] = "changed.zip"
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "Saved and embedded source manifests differ"):
            self.verify(source=False)

    def test_release_label_mapping_must_match_all_four_classes(self):
        original = self.manifest()
        for mapping in ({"0": "polite", "1": "somewhat_impolite", "2": "somewhat_polite", "3": "impolite"},
                        {"0": "impolite", "1": "somewhat_impolite", "2": "somewhat_polite"},
                        ["impolite", "somewhat_impolite", "somewhat_polite", "polite"]):
            with self.subTest(mapping=mapping):
                self.save_manifest({**original, "labels": mapping})
                with self.assertRaisesRegex(ValueError, "Release label mapping differs"):
                    self.verify(source=False)

    def test_required_release_metadata_is_checked(self):
        original = self.manifest()
        fields = ("encoding", "header", "created_at", "annotation_status", "labels", "source_manifest_sha256",
                  "source_manifest", "source_manifest_file", "original_prediction_column",
                  "unscorable_input_policy", "domains", "max_shard_bytes", "summary_sha256")
        for field in fields:
            with self.subTest(field=field):
                manifest = {key: value for key, value in original.items() if key != field}
                self.save_manifest(manifest)
                with self.assertRaisesRegex(ValueError, f"Missing release metadata: {field}"):
                    self.verify(source=False)
        for field in ("created_at", "original_prediction_column", "unscorable_input_policy"):
            with self.subTest(blank_field=field):
                self.save_manifest({**original, field: " "})
                with self.assertRaisesRegex(ValueError, f"Invalid release metadata: {field}"):
                    self.verify(source=False)

    def test_unsupported_manifest_version_is_rejected(self):
        manifest = self.manifest()
        manifest["format_version"] = 2
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "Unsupported manifest format"):
            self.verify(source=False)

    def test_recomputed_summary_hash_does_not_hide_incorrect_class_counts(self):
        path = self.output / "summary.csv"
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        rows[1][-1] = "9"
        path.write_bytes(csv_bytes(rows))
        with self.assertRaisesRegex(ValueError, "Summary SHA-256 differs"):
            self.verify(source=False)
        manifest = self.manifest()
        manifest["summary_sha256"] = verifier.sha256_file(path)
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "summary counts differ"):
            self.verify(source=False)

    def test_file_hash_and_row_counts_are_checked(self):
        manifest = self.manifest()
        info = manifest["domains"][0]["files"][0]
        info["sha256"] = "0" * 64
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            self.verify(source=False)
        info["sha256"] = verifier.sha256_file(self.output / info["path"])
        info["rows"] += 1
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "row count differs"):
            self.verify(source=False)

    def test_partial_export_is_checked_as_ordered_source_subset_and_not_complete(self):
        partial = self.root / "partial"
        self.write_release(partial, keep_labels={1, 3})
        self.verify(directory=partial)
        self.output = partial
        manifest = self.manifest()
        manifest["annotation_status"] = "complete"
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "complete export omits source rows"):
            self.verify(source=False)

    def test_original_zip_hash_is_checked_before_source_comparison(self):
        archive_size = self.archive.stat().st_size
        self.archive.write_bytes(b"X" * archive_size)
        with self.assertRaisesRegex(ValueError, "Source ZIP SHA-256 differs"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
