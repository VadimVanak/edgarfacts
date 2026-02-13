import io
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd

from edgarfacts.transforms.taxonomy.paths import construct_paths
from edgarfacts.transforms.taxonomy.reader import (
    assign_sequence,
    calculation_to_dataframe,
    read_taxonomy_arcs,
    read_taxonomy_arcs_many,
)


def _make_calculation_xml() -> str:
    return """<?xml version='1.0' encoding='UTF-8'?>
<link:linkbase xmlns:link='http://www.xbrl.org/2003/linkbase' xmlns:xlink='http://www.w3.org/1999/xlink'>
  <link:calculationLink>
    <link:calculationArc xlink:from='loc_Assets' xlink:to='loc_LiabilitiesAndStockholdersEquity' weight='1'/>
    <link:calculationArc xlink:from='http://fasb.org/us-gaap/role/statement/loc_Revenues' xlink:to='http://xbrl.us/us-gaap/role/statement/loc_NetIncomeLoss' weight='-1'/>
  </link:calculationLink>
</link:linkbase>
"""


def _make_taxonomy_zip_bytes(year: int, stmt: str = "sfp-cls") -> bytes:
    _, xml_template = construct_paths(year)
    xml_path = xml_template % (stmt, "cal")

    data = io.BytesIO()
    with zipfile.ZipFile(data, mode="w") as zf:
        zf.writestr(xml_path, _make_calculation_xml())
    return data.getvalue()


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class TaxonomyPathsTests(unittest.TestCase):
    def test_construct_paths_for_2008(self):
        zip_path, xml_template = construct_paths(2008)

        self.assertIn("XBRLUSGAAPTaxonomies-2008-03-31.zip", zip_path)
        self.assertIn("XBRLUSGAAPTaxonomies-2008-03-31/stm/", xml_template)
        self.assertIn("us-gaap-stm-ci-%s-%s-2008-03-31.xml", xml_template)

    def test_construct_paths_for_2009(self):
        zip_path, xml_template = construct_paths(2009)

        self.assertIn("XBRLUS-USGAAP-Taxonomies-2009-01-31.zip", zip_path)
        self.assertEqual(xml_template, "stm/us-gaap-stm-ci-%s-%s-2009-01-31.xml")

    def test_construct_paths_for_2022_plus(self):
        zip_path, xml_template = construct_paths(2022)

        self.assertEqual(zip_path, "https://xbrl.fasb.org/us-gaap/2022/us-gaap-2022.zip")
        self.assertEqual(xml_template, "us-gaap-2022/stm/us-gaap-stm-%s-%s-2022.xml")


class TaxonomyReaderUnitTests(unittest.TestCase):
    def test_calculation_to_dataframe_extracts_and_cleans_arcs(self):
        # Parse with ElementTree root to exercise production path
        import xml.etree.ElementTree as ET

        xml_root = ET.fromstring(_make_calculation_xml())

        result = calculation_to_dataframe(xml_root)

        self.assertListEqual(list(result.columns), ["weight", "from", "to"])
        self.assertEqual(result.loc[0, "from"], "Assets")
        self.assertEqual(result.loc[0, "to"], "LiabilitiesAndStockholdersEquity")
        self.assertEqual(result.loc[1, "from"], "Revenues")
        self.assertEqual(result.loc[1, "to"], "NetIncomeLoss")
        self.assertTrue(pd.api.types.is_float_dtype(result["weight"]))

    def test_assign_sequence_increases_for_dependencies(self):
        df = pd.DataFrame(
            {
                "from": ["A", "B", "C"],
                "to": ["B", "C", "D"],
                "weight": [1.0, 1.0, 1.0],
            }
        )

        result = assign_sequence(df)

        seq_map = dict(zip(result["from"], result["seq"]))
        self.assertGreater(seq_map["A"], seq_map["B"])
        self.assertGreater(seq_map["B"], seq_map["C"])

    @patch("edgarfacts.transforms.taxonomy.reader.urlopen")
    def test_read_taxonomy_arcs_reads_zip_and_shapes_output(self, mock_urlopen):
        mock_urlopen.return_value = _FakeResponse(_make_taxonomy_zip_bytes(2022, "sfp-cls"))

        result = read_taxonomy_arcs(2022, statements=["sfp-cls"])

        self.assertListEqual(
            list(result.columns), ["version", "statement", "seq", "from", "to", "weight"]
        )
        self.assertTrue((result["version"] == 2022).all())
        self.assertTrue((result["statement"] == "sfp-cls").all())
        self.assertEqual(len(result), 2)

    @patch("edgarfacts.transforms.taxonomy.reader.read_taxonomy_arcs")
    def test_read_taxonomy_arcs_many_sequential(self, mock_read):
        mock_read.side_effect = [
            pd.DataFrame(
                {
                    "version": [2021],
                    "statement": ["soc"],
                    "seq": [0],
                    "from": ["A"],
                    "to": ["B"],
                    "weight": [1.0],
                }
            ),
            pd.DataFrame(
                {
                    "version": [2022],
                    "statement": ["soc"],
                    "seq": [1],
                    "from": ["B"],
                    "to": ["C"],
                    "weight": [1.0],
                }
            ),
        ]

        result = read_taxonomy_arcs_many([2021, 2022], use_process_pool=False)

        self.assertEqual(len(result), 2)
        self.assertSetEqual(set(result["version"].tolist()), {2021, 2022})
        self.assertEqual(mock_read.call_count, 2)


if __name__ == "__main__":
    unittest.main()
