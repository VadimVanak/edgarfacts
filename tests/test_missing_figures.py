import io
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from edgarfacts.extract.missing_figures import read_missing_figures


US_GAAP_NS = "http://fasb.org/us-gaap/2024"


class _ClosingBytesIO(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.was_closed = False

    def close(self):
        self.was_closed = True
        super().close()


class _FakeFetcher:
    def __init__(self, payloads):
        self.payloads = payloads
        self.responses = []

    def fetch(self, url: str, ignore_exceptions: bool = False):
        payload = self.payloads.get(url)
        if payload is None:
            return None
        response = _ClosingBytesIO(payload)
        self.responses.append(response)
        return response


def _filing_summary(file_name: str = "primary.xml") -> bytes:
    return f"""
<FilingSummary>
  <InputFiles>
    <File doctype="10-Q">{file_name}</File>
  </InputFiles>
  <BaseTaxonomies>
    <BaseTaxonomy>{US_GAAP_NS}</BaseTaxonomy>
  </BaseTaxonomies>
</FilingSummary>
""".encode()


def _xbrl_instance(value: str = "123.45") -> bytes:
    return f"""
<xbrl xmlns="http://www.xbrl.org/2003/instance" xmlns:us-gaap="{US_GAAP_NS}">
  <context id="c1">
    <entity><identifier>1</identifier></entity>
    <period><startDate>2024-01-01</startDate><endDate>2024-03-31</endDate></period>
  </context>
  <unit id="u1"><measure>USD</measure></unit>
  <us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax contextRef="c1" unitRef="u1">{value}</us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax>
</xbrl>
""".encode()


class ReadMissingFiguresTests(unittest.TestCase):
    def test_malformed_submission_xml_is_logged_and_skipped(self):
        good_summary_url = (
            "https://www.sec.gov/Archives/edgar/data/1/000000000000000101/FilingSummary.xml"
        )
        bad_summary_url = (
            "https://www.sec.gov/Archives/edgar/data/1/000000000000000202/FilingSummary.xml"
        )
        good_instance_url = (
            "https://www.sec.gov/Archives/edgar/data/1/000000000000000101/primary.xml"
        )
        bad_instance_url = (
            "https://www.sec.gov/Archives/edgar/data/1/000000000000000202/primary.xml"
        )
        fetcher = _FakeFetcher(
            {
                good_summary_url: _filing_summary(),
                bad_summary_url: _filing_summary(),
                good_instance_url: _xbrl_instance(),
                bad_instance_url: b"<xbrl>truncated\x00",
            }
        )
        logger = Mock()
        sub = pd.DataFrame({"cik": [1, 1], "adsh": [101, 202]})

        result = read_missing_figures(
            sub,
            np.array(["RevenueFromContractWithCustomerExcludingAssessedTax"]),
            fetcher,
            logger,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["adsh"].tolist(), [101])
        warning_messages = [call.args[0] for call in logger.warning.call_args_list]
        self.assertTrue(
            any(
                message.startswith(
                    f"Skipping submission 202: failed to parse XML from {bad_instance_url}:"
                )
                for message in warning_messages
            )
        )
        self.assertTrue(all(response.was_closed for response in fetcher.responses))

    def test_malformed_filing_summary_is_logged_and_skipped(self):
        summary_url = (
            "https://www.sec.gov/Archives/edgar/data/1/000000000000000303/FilingSummary.xml"
        )
        fetcher = _FakeFetcher({summary_url: b"<FilingSummary><InputFiles>"})
        logger = Mock()
        sub = pd.DataFrame({"cik": [1], "adsh": [303]})

        result = read_missing_figures(
            sub,
            np.array(["RevenueFromContractWithCustomerExcludingAssessedTax"]),
            fetcher,
            logger,
        )

        self.assertIsNone(result)
        warning_messages = [call.args[0] for call in logger.warning.call_args_list]
        self.assertTrue(
            any(
                message.startswith(
                    f"Skipping submission 303: failed to parse XML from {summary_url}:"
                )
                for message in warning_messages
            )
        )
        logger.warning.assert_any_call("No main submission file found for submission 303")
        self.assertTrue(all(response.was_closed for response in fetcher.responses))


if __name__ == "__main__":
    unittest.main()
