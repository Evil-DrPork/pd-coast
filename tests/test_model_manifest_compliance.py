import tempfile
import unittest
from pathlib import Path

from poker44.utils.model_manifest import _sha256_for_files, evaluate_manifest_compliance


class ModelManifestComplianceTests(unittest.TestCase):
    def test_implementation_hash_is_checkout_and_line_ending_portable(self) -> None:
        with (
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
        ):
            first = Path(first_dir)
            second = Path(second_dir)
            for root, newline in ((first, b"\r\n"), (second, b"\n")):
                package = root / "package"
                package.mkdir()
                (package / "a.py").write_bytes(newline.join((b"alpha", b"beta", b"")))
                (package / "b.txt").write_bytes(newline.join((b"one", b"two", b"")))

            first_hash = _sha256_for_files(
                [first / "package" / "b.txt", first / "package" / "a.py"],
                repo_root=first,
            )
            second_hash = _sha256_for_files(
                [second / "package" / "a.py", second / "package" / "b.txt"],
                repo_root=second,
            )
            self.assertEqual(first_hash, second_hash)

    def test_reference_miner_manifest_can_use_reference_repo(self) -> None:
        manifest = {
            "open_source": True,
            "repo_url": "https://github.com/Poker44/Poker44-subnet",
            "repo_commit": "69df8454943e965e08d3b7923182e0a69981584c",
            "model_name": "poker44-reference-heuristic",
            "model_version": "1",
            "training_data_statement": "Reference heuristic miner. No training.",
            "private_data_attestation": "No validator-private data used.",
            "implementation_files": ["neurons/miner.py"],
            "implementation_sha256": "abc123",
        }

        compliance = evaluate_manifest_compliance(manifest)
        self.assertEqual(compliance["status"], "transparent")
        self.assertEqual(compliance["missing_fields"], [])
        self.assertEqual(compliance["policy_violations"], [])

    def test_custom_model_using_reference_repo_is_not_transparent(self) -> None:
        manifest = {
            "open_source": True,
            "repo_url": "https://github.com/Poker44/Poker44-subnet",
            "repo_commit": "69df8454943e965e08d3b7923182e0a69981584c",
            "model_name": "poker44-rf-bot-detector",
            "model_version": "v4",
            "training_data_statement": "Custom model.",
            "private_data_attestation": "No validator-private data used.",
            "implementation_files": ["neurons/miner.py"],
            "implementation_sha256": "abc123",
        }

        compliance = evaluate_manifest_compliance(manifest)
        self.assertEqual(compliance["status"], "opaque")
        self.assertIn("repo_url_must_point_to_model_repo", compliance["policy_violations"])

    def test_placeholder_commit_is_not_transparent(self) -> None:
        manifest = {
            "open_source": True,
            "repo_url": "https://github.com/example/poker44-custom",
            "repo_commit": "<full_git_commit_sha>",
            "model_name": "TBM",
            "model_version": "1.0",
            "training_data_statement": "Custom model.",
            "private_data_attestation": "No validator-private data used.",
            "implementation_files": ["neurons/miner.py"],
            "implementation_sha256": "abc123",
        }

        compliance = evaluate_manifest_compliance(manifest)
        self.assertEqual(compliance["status"], "opaque")
        self.assertIn("repo_commit_invalid", compliance["policy_violations"])

    def test_missing_implementation_details_is_not_transparent(self) -> None:
        manifest = {
            "open_source": True,
            "repo_url": "https://github.com/example/poker44-custom",
            "repo_commit": "69df8454943e965e08d3b7923182e0a69981584c",
            "model_name": "TBM",
            "model_version": "1.0",
            "training_data_statement": "Custom model.",
            "private_data_attestation": "No validator-private data used.",
        }

        compliance = evaluate_manifest_compliance(manifest)
        self.assertEqual(compliance["status"], "opaque")
        self.assertIn("implementation_files", compliance["missing_fields"])
        self.assertIn("implementation_sha256", compliance["missing_fields"])


if __name__ == "__main__":
    unittest.main()
