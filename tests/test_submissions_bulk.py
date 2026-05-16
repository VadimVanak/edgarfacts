import unittest
from http.client import IncompleteRead
from unittest.mock import Mock

import pandas as pd

from edgarfacts.extract.submissions_bulk import update_version_info


class _IncompleteResponse:
    def __init__(self, partial: bytes):
        self.partial = partial
        self.was_closed = False

    def read(self, _size=-1):
        raise IncompleteRead(self.partial)

    def close(self):
        self.was_closed = True


class _ChunkedResponse:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.was_closed = False

    def read(self, _size=-1):
        if self.chunks:
            chunk = self.chunks.pop(0)
            if isinstance(chunk, Exception):
                raise chunk
            return chunk
        return b""

    def close(self):
        self.was_closed = True


class _FakeFetcher:
    def __init__(self, response):
        self.response = response

    def fetch(self, url: str, ignore_exceptions: bool = False):
        return self.response


class UpdateVersionInfoTests(unittest.TestCase):
    def _submission(self):
        return pd.DataFrame(
            {
                "cik": [1],
                "adsh": [101],
                "file": ["primary.htm"],
            }
        )

    def test_incomplete_response_partial_bytes_are_scanned(self):
        response = _IncompleteResponse(b'<xbrli xmlns:us-gaap="http://fasb.org/us-gaap/2024">')
        logger = Mock()

        result = update_version_info(self._submission(), _FakeFetcher(response), logger)

        self.assertEqual(result["version"].tolist(), [2024])
        self.assertTrue(response.was_closed)
        logger.warning.assert_called_once()

    def test_truncated_response_without_version_does_not_abort(self):
        response = _ChunkedResponse([b"<html>", IncompleteRead(b"truncated")])
        logger = Mock()

        result = update_version_info(self._submission(), _FakeFetcher(response), logger)

        self.assertEqual(result["version"].tolist(), [0])
        self.assertTrue(response.was_closed)
        logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
