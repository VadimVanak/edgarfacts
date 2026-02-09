"""
Low-level I/O helpers for taxonomy processing.

This module is intentionally minimal and isolated:
- network access
- ZIP handling
- XML parsing

Higher-level logic (iteration over statements / years, sequencing, etc.)
lives in reader.py and calcs.py.
"""

from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from typing import Iterable, Dict


def fetch_zip(url: str) -> ZipFile:
    """
    Download a ZIP file from a URL and return a ZipFile object.

    Parameters
    ----------
    url:
        HTTP(S) URL of the ZIP archive.

    Returns
    -------
    zipfile.ZipFile
        Open ZIP archive backed by an in-memory buffer.

    Notes
    -----
    - Caller is responsible for closing the ZipFile if needed.
    - No retries or rate-limiting are applied here by design.
    """
    resp = urlopen(Request(url))
    data = resp.read()
    resp.close()
    return ZipFile(BytesIO(data))


def read_xml_from_zip(zf: ZipFile, internal_path: str) -> ElementTree.Element:
    """
    Read and parse an XML file from an open ZipFile.

    Parameters
    ----------
    zf:
        Open ZipFile instance.
    internal_path:
        Path inside the ZIP archive.

    Returns
    -------
    xml.etree.ElementTree.Element
        Root element of the parsed XML document.
    """
    with zf.open(internal_path) as fh:
        return ElementTree.parse(fh).getroot()


def safe_open_many(zf: ZipFile, paths: Iterable[str]) -> Dict[str, ElementTree.Element]:
    """
    Attempt to read multiple XML files from a ZIP archive.

    Missing files are silently skipped.

    Parameters
    ----------
    zf:
        Open ZipFile.
    paths:
        Iterable of internal ZIP paths.

    Returns
    -------
    dict
        Mapping {path -> XML root} for successfully read files.
    """
    out: Dict[str, ElementTree.Element] = {}
    for p in paths:
        try:
            out[p] = read_xml_from_zip(zf, p)
        except KeyError:
            # file not present in this taxonomy version
            continue
    return out
