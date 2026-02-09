"""
Path construction for US-GAAP taxonomy packages.

We support:
- 2008 taxonomy hosted at taxonomies.xbrl.us with a special folder layout
- 2009 taxonomy hosted at taxonomies.xbrl.us with a different layout
- 2011+ taxonomies hosted at xbrl.fasb.org (note: 2010 is typically not available in the same way)

The returned `xml_path_template` is a printf-style template requiring two substitutions:
  (statement_code, linkbase_kind)

Example:
  xml_path_template % ("sfp-cls", "cal")
"""

from __future__ import annotations

from typing import Tuple


def construct_paths(year: int) -> Tuple[str, str]:
    """
    Construct download URL (zip_path) and internal XML path template (xml_path_template)
    for the specified taxonomy year.

    Parameters
    ----------
    year:
        Taxonomy year (e.g., 2008, 2009, 2011..).

    Returns
    -------
    (zip_path, xml_path_template)
        zip_path: URL to taxonomy ZIP package
        xml_path_template: printf-style template for statement files inside the ZIP

    Notes
    -----
    Historical naming conventions:
    - year < 2012 uses "ci-" statement prefix in file names
    - 2008 uses suffix "-03-31"
    - years < 2022 (except 2008) use suffix "-01-31"
    - 2022+ uses no suffix
    """
    prefix = "ci-" if year < 2012 else ""
    suffix = "-03-31" if year == 2008 else ("-01-31" if year < 2022 else "")

    if year == 2008:
        zip_path = (
            "https://taxonomies.xbrl.us/us-gaap/1.0/doc/"
            f"XBRLUSGAAPTaxonomies-{year}{suffix}.zip"
        )
        xml_path_template = (
            f"XBRLUSGAAPTaxonomies-{year}{suffix}/stm/"
            f"us-gaap-stm-{prefix}%s-%s-{year}{suffix}.xml"
        )
    elif year == 2009:
        zip_path = (
            "https://taxonomies.xbrl.us/us-gaap/2009/doc/"
            f"XBRLUS-USGAAP-Taxonomies-{year}{suffix}.zip"
        )
        xml_path_template = f"stm/us-gaap-stm-{prefix}%s-%s-{year}{suffix}.xml"
    else:
        zip_path = f"https://xbrl.fasb.org/us-gaap/{year}/us-gaap-{year}{suffix}.zip"
        xml_path_template = (
            f"us-gaap-{year}{suffix}/stm/"
            f"us-gaap-stm-{prefix}%s-%s-{year}{suffix}.xml"
        )

    return zip_path, xml_path_template
