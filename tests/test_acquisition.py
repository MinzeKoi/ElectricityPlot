"""Exercise bounded-download and provenance failures without a network service."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.data.acquire import fetch_refit
from virtual_submeter.data.common import DataError, read_json, sha256


class Response(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {}


class AcquisitionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "raw"
        self.data = b"Time,Unix,Aggregate,Appliance1,Issues\n" + b"2013-01-01 00:00:00,1,100,10,0\n" * 100
        self.metadata = {"metadata": {"license": {"id": "cc-by-4.0"}}, "files": [
            {"key": "CLEAN_House2.csv", "size": len(self.data), "checksum": "md5:" + hashlib.md5(self.data).hexdigest()}
        ]}

    def api(self):
        return Response(json.dumps(self.metadata).encode())

    def test_real_prefix_receipt_does_not_claim_full_publisher_verification(self):
        with patch("virtual_submeter.data.acquire.urlopen", side_effect=[self.api(), Response(self.data[:1024], 206, {"Content-Range": f"bytes 0-1023/{len(self.data)}"})]) as request:
            result = fetch_refit(self.directory, ["2"], 1024)
        self.assertEqual(request.call_args_list[1].args[0].get_header("Range"), "bytes=0-1023")
        receipt = result["files"][0]
        self.assertEqual(receipt["kind"], "real_prefix")
        self.assertFalse(receipt["publisher_checksum_verified"])
        path = self.directory / "CLEAN_House2.csv"
        self.assertTrue(path.read_bytes().endswith(b"\n"))
        self.assertEqual(receipt["sha256"], sha256(path))
        self.assertLessEqual(path.stat().st_size, 1024)

    def test_complete_download_checks_actual_publisher_digest(self):
        with patch("virtual_submeter.data.acquire.urlopen", side_effect=[self.api(), Response(self.data)]):
            fetch_refit(self.directory, ["2"], 0)
        receipt = read_json(self.directory / "CLEAN_House2.receipt.json")
        self.assertTrue(receipt["publisher_checksum_verified"])
        self.assertEqual(receipt["kind"], "real_complete")

    def test_ignored_range_and_truncated_prefix_do_not_leave_success_files(self):
        for response in (Response(self.data, 200), Response(self.data[:100], 206, {"Content-Range": f"bytes 0-1023/{len(self.data)}"})):
            with self.subTest(response=response.status):
                with patch("virtual_submeter.data.acquire.urlopen", side_effect=[self.api(), response]), self.assertRaises(DataError):
                    fetch_refit(self.directory, ["2"], 1024)
                self.assertFalse((self.directory / "CLEAN_House2.csv").exists())
                self.assertFalse((self.directory / "CLEAN_House2.receipt.json").exists())
                self.assertFalse((self.directory / "CLEAN_House2.download").exists())

    def test_wrong_full_checksum_and_unknown_license_are_rejected(self):
        with patch("virtual_submeter.data.acquire.urlopen", side_effect=[self.api(), Response(self.data.replace(b"100", b"999"))]), self.assertRaisesRegex(DataError, "checksum"):
            fetch_refit(self.directory, ["2"], 0)
        self.assertFalse((self.directory / "CLEAN_House2.csv").exists())
        self.metadata["metadata"]["license"]["id"] = "unknown"
        with patch("virtual_submeter.data.acquire.urlopen", side_effect=[self.api()]), self.assertRaisesRegex(DataError, "license"):
            fetch_refit(self.directory, ["2"], 1024)


if __name__ == "__main__":
    unittest.main()
