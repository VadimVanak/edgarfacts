import numpy as np
import pandas as pd
import os
import sys
import logging
import time
import ssl
import tempfile
import re
from zipfile import ZipFile
from io import BytesIO
from urllib.request import Request, urlopen
from typing import List, Optional
from multiprocessing.pool import Pool
from functools import partial
from xml.etree import ElementTree
import msgspec

def get_logger():
    """
    Configures and returns a logger instance for logging messages.

    This function sets up a logger that outputs log messages to stderr.
    The logger uses a simple format that outputs only the message text.
    If the logger already has handlers, it avoids adding duplicate handlers.
    The logging level is set to INFO.

    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    logger = logging.getLogger(__name__)  

    # Check if the logger already has any handlers to avoid adding duplicate handlers
    if len(logger.handlers) == 0:
        c_handler = logging.StreamHandler(sys.stdout)  
        c_format = logging.Formatter("%(message)s")  # Output just the message)
        c_handler.setFormatter(c_format)  
        logger.addHandler(c_handler)  
    
    logger.setLevel(logging.INFO)  # Set the logging level to INFO
    return logger  
	

class URLFetcher:
    def __init__(self, logger):
        """
        Initialize the URLFetcher with a given EDGAR username and a logger instance.

        Parameters:
        - edgar_username (str): The username to use when making requests to the SEC EDGAR system.
        - logger (logging.Logger): A logger instance to log warnings and other messages.
        """
        self._edgar_username = self._get_edgar_username()
        # Set initial request time to enforce the request rate limit.
        self._prev_req_time = time.time() - 0.11
        self._logger = logger

    def _get_edgar_username(self):
        """
        Retrieves the EDGAR username from a secret management service.

        This function attempts to fetch the EDGAR username using two different methods,
        depending on the environment:
        1. In a Kaggle environment, it uses the `kaggle_secrets` module to retrieve the secret.
        2. In a Google Colab environment, it uses the `google.colab` module to get the username.

        Returns:
            edgar_username (str): The EDGAR username retrieved, or None if not found.
        """
        try:
            from kaggle_secrets import UserSecretsClient

            edgar_username = UserSecretsClient().get_secret("edgar_username")
        except:
            from google.colab import userdata

            edgar_username = userdata.get("edgar_username")
        return edgar_username

    def _limit_request_ratio(self):
        """
        Enforce a minimum time interval between consecutive requests.

        This method ensures that there is at least a 0.11-second gap between requests
        to comply with SEC EDGAR's rate limits.
        """
        time.sleep(max(0.11 - time.time() + self._prev_req_time, 0.001))
        self._prev_req_time = time.time()

    def fetch(self, url, ignore_exceptions=False):
        """
        Fetch the content from the specified URL.

        Parameters:
        - url (str): The URL to fetch.
        - ignore_exceptions (bool): If True, the method will return None in case of an error.
          Otherwise, it will raise an exception.

        Returns:
        - HTTPResponse object if the request is successful,
          or None if an exception occurs and ignore_exceptions is set to True.
        """
        req = Request(url)

        if url.startswith(("https://www.sec.gov","https://data.sec.gov")):
            # If the URL is for the SEC EDGAR system, add the "User-Agent" header with
            # the provided EDGAR username and ensure that the request rate limit is respected.
            req.add_header("User-Agent", self._edgar_username)
            self._limit_request_ratio()

        if not ignore_exceptions:
            return urlopen(req)

        try:
            resp = urlopen(req)
        except Exception as e:
            # If an exception occurs and ignore_exceptions is True, log a warning and return None.
            self._logger.warning(f"Failed to fetch {url}: {e}")
            return None

        return resp
		

def read_tickers(fetcher):
    # The sec.gov website assigns each company a unique Central Index Key (CIK) for identification.
    # This function downloads a mapping of CIKs to more commonly used ticker symbols.
    # The ticker symbols are obtained from the URL: https://www.sec.gov/include/ticker.txt.
    with fetcher.fetch("https://www.sec.gov/include/ticker.txt") as resp:
        # Load the data into a pandas DataFrame,
        tickers = pd.read_csv(BytesIO(resp.read()), sep='\t', names=['ticker', 'cik'])
    
    tickers = tickers[tickers['ticker'].notnull()].drop_duplicates(subset='cik')
    
    # To optimize memory usage, convert the 'cik' column to an unsigned 32-bit integer type.
    tickers['cik'] = tickers['cik'].astype(np.uint32)
    
    # Convert the 'ticker' column to a pandas Categorical type, 
    # which is more memory-efficient for string data.
    tickers['ticker'] = pd.Categorical(tickers['ticker'])
    
    return tickers
	
	
def read_tags(fetcher):
    tag_list = np.array([])
    for year in range(2012,2025):
        suffix = "-01-31" if year < 2022 else ""
        zip_path = f"https://xbrl.fasb.org/us-gaap/{year}/us-gaap-{year}{suffix}.zip"
        xsd_path = f"us-gaap-{year}{suffix}/elts/us-gaap-{year}{suffix}.xsd"   
        
        with fetcher.fetch(zip_path) as resp:
            zf = ZipFile(BytesIO(resp.read()))
        with zf.open(xsd_path) as x:
            root = ElementTree.parse(x).getroot()      
        tag_df = pd.DataFrame(
            [r.attrib for r in root.iter("{http://www.w3.org/2001/XMLSchema}element")]
        ).query(
            "type.isin(['xbrli:monetaryItemType','xbrli:sharesItemType']) and abstract.isnull()"
        )
        tag_list = np.union1d(tag_list, tag_df['name'])
    return np.union1d(tag_list, ["EntityCommonStockSharesOutstanding", "EntityPublicFloat"])
	
	
def read_periods(fetcher):
    # Download the list of available financial statement data sets from the SEC's website.
    # The data sets are available starting from Q1 2009.
    with fetcher.fetch(
        "https://www.sec.gov/dera/data/financial-statement-data-sets"
    ) as resp:
        page = resp.read().decode("utf-8")

    # Use a regular expression to find all matching financial statement data sets.
    # The regex looks for patterns like
    # "/files/dera/data/financial-statement-data-sets/YYYYqQQ.zip".
    matches = re.findall(
        "\/files\/dera\/data\/financial-statement-data-sets\/(\d{4}q[1-4])\.zip", page
    )
    # Convert the matched strings into tuples of integers (year, quarter)
    return [tuple(map(int, m.split("q"))) for m in np.sort(matches)]
	
	
def read_submissions(period, fetcher, valid_ciks, logger):
    # This function retrieves and processes submission information for a specific period
    # from the SEC's financial statement data sets.

    # Extract the year and quarter from the period tuple.
    year, quarter = period

    # Log the start of the data loading process for the specified year and quarter.
    logger.info(f"Loading year {year} quarter {quarter}")

    # Create a request to download the zipped financial statement data for the given period.
    with fetcher.fetch(
        f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{year}q{quarter}.zip"
    ) as resp:
        zf = ZipFile(BytesIO(resp.read()))

    # Read the 'sub.txt' file from the zip, which contains submission details.
    sub = pd.read_csv(
        zf.open("sub.txt"),
        sep="\t",
        usecols=[
            "adsh",  # Accession Number (unique identifier for the submission)
            "cik",  # Central Index Key (unique identifier for companies)
            "sic",  # Standard Industrial Classification code
            "form",  # Form type (e.g., 10-Q, 10-K)
            "period",  # Reporting period
            "accepted",  # Date the submission was accepted by the SEC
            "instance",  # Submission filename
        ],
    )

    # Filter the submissions to include only specific form types (10-Q, 10-K, and their amendments),
    # non-null periods, and valid CIKs.
    sub = sub[
        sub["form"].isin(["10-Q", "10-K", "10-Q/A", "10-K/A"])
        & sub["period"].notnull()
        & sub["cik"].isin(valid_ciks)
    ].rename(columns={"instance": "file"})

    # Clean and convert the 'adsh' (Accession Number) to an integer.
    sub["adsh"] = sub["adsh"].str.replace("-", "").astype(int)

    # Convert the 'sic' code to an integer, filling any missing values with 0.
    sub["sic"] = sub["sic"].astype(float).fillna(0).astype(int)

    # Load the 'num.txt' file to get the US-GAAP taxonomy version used in each submission.
    num_df = pd.read_csv(zf.open("num.txt"), sep="\t", usecols=["adsh", "version"])

    # Filter the rows to include only those related to the US-GAAP taxonomy,
    # and extract the year from the version string.
    num_df = num_df[num_df["version"].str.startswith("us-gaap")]
    num_df["version"] = num_df["version"].str.split("/").str[1].str[0:4].astype(int)

    # Clean and convert the 'adsh' (Accession Number) to an integer.
    num_df["adsh"] = num_df["adsh"].str.replace("-", "").astype(int)

    # Remove duplicate rows to ensure unique combinations of 'adsh' and 'version'.
    num_df = num_df.drop_duplicates()

    # Merge the submission data with the taxonomy version information based on the 'adsh' key.
    sub = sub.merge(num_df, how="left", on="adsh")
    sub["version"] = sub["version"].fillna(0).astype(int)

    # Convert the 'period' and 'accepted' columns to datetime objects for easier manipulation.
    sub["period"] = pd.to_datetime(sub["period"].astype(int), format="%Y%m%d").astype(
        "datetime64[s]"
    )
    sub["accepted"] = sub["accepted"].astype("datetime64[s]")

    return sub


# Parallel processing function
def read_submissions_parallel(period_arr, fetcher, valid_ciks, logger):
    # This function processes submission data in parallel for multiple periods 
    # to improve efficiency.

    # Create a partial function with fixed parameters
    partial_read_submissions = partial(
        read_submissions,
        fetcher = fetcher,
        valid_ciks = valid_ciks,
        logger = logger,
    )

    # Initialize a pool of worker processes for parallel execution.
    pool = Pool()

    # Map the partial_read_submissions function to each period in the period_arr.
    results = pool.map(partial_read_submissions, period_arr)

    # Close the pool and wait for all worker processes to complete.
    pool.close()
    pool.join()

    # Concatenate the results from all periods into a single DataFrame and return it.
    return pd.concat(results)
	
	
# These classes define the structure of the company facts JSON file for parsing.

class TagItem(msgspec.Struct):
    end: str                     # The end date of the reporting period.
    accn: str                    # The accession number, a unique ID for the submission.
    val: float                   # The reported value for the financial metric.
    form: str                    # The form type (e.g., 10-K, 10-Q).
    start: Optional[str] = None  # The start date of the reporting period (optional).

    def to_tuple(self, tag):
        # Convert the TagItem object into a tuple 
        return (
            int(self.accn.replace("-", "", 2)),  # Cleaned and converted accession number.
            tag,  # The financial metric tag.
            (
                self.end
                if self.start is None  # Use the end date if the start date is not provided.
                else "1900-01-01" if self.start[0:4] < "1900" else self.start  # invalid start date
            ),
            self.end,  # The end date of the reporting period.
            self.val,  # The reported value.
        )

class Units(msgspec.Struct):
    USD: List[TagItem] = None     # List of TagItems reported in USD.
    shares: List[TagItem] = None  # List of TagItems reported in shares.

    def to_list(self, tag):
        # Convert the Units object into a list of tuples 
        l = []
        if self.USD is not None:
            l += [
                t.to_tuple(tag)
                for t in self.USD
                if t.form in ["10-K", "10-Q", "10-K/A", "10-Q/A"]  
            ]
        if self.shares is not None:
            l += [
                t.to_tuple(tag)
                for t in self.shares
                if t.form in ["10-K", "10-Q", "10-K/A", "10-Q/A"] 
            ]
        return l

class Tag(msgspec.Struct):
    units: Units  # Contains financial data categorized by units (USD and shares are supported).

class Dei(msgspec.Struct):  # Document Entity Information
    EntityCommonStockSharesOutstanding: Optional[Tag] = None  # Number of shares outstanding.
    EntityPublicFloat: Optional[Tag] = None                   # Data related to public float.

    def to_list(self):
        # Convert the Dei object into a list of tuples 
        l = []
        if self.EntityCommonStockSharesOutstanding is not None:
            l += self.EntityCommonStockSharesOutstanding.units.to_list(
                "EntityCommonStockSharesOutstanding"
            )
        if self.EntityPublicFloat is not None:
            l += self.EntityPublicFloat.units.to_list("EntityPublicFloat")
        return l

# Get a decoder for the company facts JSON file, 
# focusing on specific tags passed in the tag_list parameter.
def get_decoder(tag_list):
    # Define the structure of the US-GAAP section of the JSON based on the tags of interest.
    tag_types = [(t, Tag | None, None) for t in tag_list]
    UsGaap = msgspec.defstruct("UsGaap", tag_types)

    class Facts(msgspec.Struct):
        us_gaap: Optional[UsGaap] = msgspec.field(name="us-gaap", default=None)
        dei: Optional[Dei] = None  # DEI (Document Entity Information) section of the JSON.

        def to_dataframe(self):
            # Convert the Facts object into a dataframe.
            l = []
            if self.us_gaap is not None:
                # Extract and process financial data under US-GAAP.
                attr = [a for a in dir(self.us_gaap) if not a.startswith("__")]
                for a in attr:
                    item = getattr(self.us_gaap, a)
                    if item is not None:
                        l += item.units.to_list(a)
            if self.dei is not None:
                # Extract and process DEI (Document Entity Information) data.
                l += self.dei.to_list()
            df = pd.DataFrame(l, columns=["adsh", "tag", "start", "end", "value"])
            df["adsh"] = df["adsh"].astype(np.int64)
            df["tag"] = pd.Categorical(df["tag"], categories=self._tag_list)
            df["start"] = df["start"].astype("datetime64[s]")
            df["end"] = df["end"].astype("datetime64[s]")
            return df

    # Store the tag map in the Facts class 
    Facts._tag_list = tag_list 
                                       
    class Figures(msgspec.Struct):
        facts: Facts = None  # The core financial data structure.

        def to_dataframe(self):
            # Convert the Figures object to a NumPy array.
            return self.facts.to_dataframe()

    # Return a JSON decoder configured to parse Figures.
    return msgspec.json.Decoder(Figures)

# Extract figures from the companyfacts.zip file.
def load_facts(valid_ciks, tag_list, fetcher, logger):
    # Get the decoder configured for specific tags.
    decoder = get_decoder(tag_list)
    df_array = []

    # Send a request to download the company facts data from the SEC.
    with fetcher.fetch(
        "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
    ) as resp:
        zf = ZipFile(BytesIO(resp.read()))

    # Process the downloaded zip file.
    # Filter the file list to include only those matching the valid CIKs.
    nlist = [n for n in zf.namelist() if int(n[3:13]) in valid_ciks]

    # Iterate over the filtered files and decode their content.
    for i, file in enumerate(nlist):
        if i == 0 or i % 1000 == 999:
            logger.info(f"Processing file {i+1} of {len(nlist)}")
        with zf.open(file) as f:
            content = f.read()
        d = decoder.decode(content)
        if d.facts is not None:
            df = d.to_dataframe()
            if len(df)> 0:
                df_array += [df]

    return pd.concat(df_array)
	
	
# Define a class representing recent filings with various attributes
class Recent(msgspec.Struct):
    form: List[str] = []               # List of form types (e.g., 10-Q, 10-K)
    accessionNumber: List[str] = []    # List of accession numbers for the filings
    reportDate: List[str] = []         # List of report dates for the filings
    primaryDocument: List[str] = []    # List of primary document filenames
    acceptanceDateTime: List[str] = [] # List of acceptance dates and times

# Define a class representing filings, containing recent filings information
class Filings(msgspec.Struct):
    recent: Recent                     # Recent filings data

# Define a class representing submissions, which includes company identifiers and filings
class Submissions(msgspec.Struct):
    cik: Optional[str] = None          # Central Index Key (CIK) of the company
    sic: Optional[str] = None          # Standard Industrial Classification (SIC)
                                       # code of the company
    filings: Optional[Filings] = None  # Filings information for the company

    # Convert the submissions data to a pandas DataFrame for further analysis
    def to_dataframe(self):
        return pd.DataFrame(
            {
                "adsh": np.char.replace(
                    self.filings.recent.accessionNumber, "-", ""
                ).astype("int64"),  # Clean and convert accession numbers to integers
                "cik": int(self.cik),  
                # Convert SIC to integer, default to 0 if not available
                "sic": 0 if len(self.sic) == 0 else int(self.sic),  
                "form": self.filings.recent.form,  # Form types (e.g., 10-Q, 10-K)
                "period": np.array(self.filings.recent.reportDate).astype(
                    "datetime64[D]"
                ),  # Convert report dates to datetime
                "accepted": np.char.replace(
                    self.filings.recent.acceptanceDateTime, ".000Z", ""
                ).astype("datetime64[s]"),  # Convert acceptance times to datetime
                "file": self.filings.recent.primaryDocument,  # Primary document filenames
            }
        ).query("form.isin(['10-Q','10-K','10-Q/A','10-K/A'])")  # Filter for specific form types

# Function to read and process submissions data for a list of valid CIKs
def read_submissions_2(valid_ciks, fetcher, logger):
    decoder = msgspec.json.Decoder(Submissions)  

    df_array = []  # List to store DataFrames from each submission

    # Send a request to download the bulk submissions data from the SEC
    with fetcher.fetch(
        "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
    ) as resp:
        zf = ZipFile(BytesIO(resp.read()))

    # Filter the ZIP file contents to include only those with valid CIKs
    nlist = [
        n for n in zf.namelist() if n[3:13].isdigit() and int(n[3:13]) in valid_ciks
    ]
    for i, file in enumerate(nlist):  # Iterate over the filtered file list
        if i == 0 or i % 1000 == 999:  # Log progress every 1000 files
            logger.info(f"Processing file {i+1} of {len(nlist)}")
        with zf.open(file) as f:  # Open each file in the ZIP archive
            d = decoder.decode(f.read())  
        try:
            # Check if the filings have any accession numbers
            _ = len(d.filings.recent.accessionNumber)  
        except:  # If there's an error (e.g., missing data), pass
            continue
        # If the data is valid, convert it to a DataFrame 
        df_array += [d.to_dataframe()]

    return pd.concat(df_array)
	
	
# The version information is not available in the files extracted from submissions.zip.
# To obtain this information, we need to read each individual submission file.
def update_version_info(sub2, fetcher, logger):
    sub2 = sub2.copy().reset_index(drop=True)
    # Initialize a new column 'version' to store the extracted version information
    sub2["version"] = 0

    # Iterate over each row in the DataFrame to fetch and update version information
    for index, row in sub2.iterrows():

        # Log progress every 100 submissions
        if index == 0 or index % 100 == 99:
            logger.info(f"Loading version info {index+1} of {len(sub2)}")

        # Construct the URL for the submission file based on the CIK and accession number (adsh)
        response = fetcher.fetch(
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{row['cik']}/{row['adsh']:018d}/{row['file']}",
            ignore_exceptions=True,
        )
        if response is None:
            continue
        version = 0
        previous_chunk = ""
        # Read the response content in chunks
        while True:
            chunk = response.read(4096).decode("utf-8")
            if not chunk: # End of file
                response.close()
                break  
            combined_chunk = previous_chunk + chunk
            match = re.search(r"us-gaap\/(20\d{2})", combined_chunk)
            if match:
                response.close()
                version = int(match.group(1))
                break
            previous_chunk = chunk
        sub2.loc[index, "version"] = version


    # Drop the 'file' column as it's no longer needed and return the updated DataFrame
    return sub2.drop(columns="file")


# For submissions that have been amended by a later submission, set an amendment flag
# and record the accession number of the latest amendment.
def set_amended_flag(sub):
    # Identify the latest submission for each CIK and reporting period,
    # and set the 'amendment_adsh' to 0 for the latest submission,
    # and to the latest accession number for earlier ones.
    sub["amendment_adsh"] = (
        sub.sort_values(by=["cik", "period", "accepted"])
        .groupby(["cik", "period"])["adsh"]
        .transform(
            lambda x: np.where(x == x.iloc[-1], 0, x.iloc[-1])
        )  # Update non-latest submissions with latest adsh
    )

    # Create a boolean column to indicate if a submission has been amended
    sub["is_amended"] = sub["amendment_adsh"] != 0

    return sub


def read_missing_submissions(missing_adsh, fetcher):
    # We will try to extract the CIK from the ADSH number.
    # The first part of the ADSH number does not always contain the CIK,
    # but we have no other information.
    ciks = np.unique(missing_adsh // 100_000_000)
    decoder = msgspec.json.Decoder(Submissions)
    df_list = []
    for c in ciks:
        response = fetcher.fetch(
            f"https://data.sec.gov/submissions/CIK{c:010d}.json", ignore_exceptions=True
        )
        if response is not None:
            d = decoder.decode(response.read())
            response.close()
            df_list += [d.to_dataframe()]
    return (
        pd.concat(df_list).query("adsh.isin(@missing_adsh)")
        if len(df_list) > 0
        else None
    )
	
	
def extract_submissions_and_facts_internal(fetcher, logger, debug_mode=False):
    # Load the mapping of ticker symbols to CIKs from the SEC's website
    tickers = read_tickers(fetcher)
    logger.info(f"{len(tickers)} tickers loaded")

    # Retrieve the list of available periods (quarters) for which financial data is available
    period_arr = read_periods(fetcher)
    logger.info(f"Last available period is {period_arr[-1]}")

    # DEBUG MODE
    # If debug_mode is enabled, limit the data to only two companies and two periods
    # This significantly reduces runtime, making it useful for debugging purposes
    if debug_mode:
        tickers = tickers.query(
            "ticker=='msft' or ticker=='nvda'"
        )  # Filter for Microsoft and NVIDIA
        period_arr = period_arr[-2:]  # Use only the last two periods
    # END DEBUG MODE

    # Load the list of US-GAAP tags
    tag_list = read_tags(fetcher)

    # Extract the unique CIKs (company identifiers) from the tickers data
    valid_ciks = tickers.cik.unique()

    # Load financial facts (e.g., revenues, assets) for the valid CIKs and available tags
    df = load_facts(valid_ciks, tag_list, fetcher, logger)
    logger.info("Company facts loaded")

    # Load submission data using parallel processing
    sub = read_submissions_parallel(period_arr, fetcher, valid_ciks, logger)

    # Load additional submission data from submissions.zip
    sub2 = read_submissions_2(valid_ciks, fetcher, logger)

    # There are more submissions in submissions.zip because this file is updated daily
    # but it has no version information. We only want to read version information
    # for submissions where we don't have it, so we remove such submissions from sub2
    sub2 = sub2[~sub2["adsh"].isin(sub[(sub["version"] != 0)]["adsh"])]
    logger.info("Submissions loaded")

    sub2 = pd.concat(
        (
            sub2,
            sub[(sub["version"] == 0) & ~sub["adsh"].isin(sub2["adsh"])].drop(
                columns="version"
            ),
        )
    )
    # Filter submissions in sub2 that are not in sub and are present in df,
    # then update with version information
    sub2 = sub2[
        ~sub2["adsh"].isin(sub[sub["version"] != 0]["adsh"])
        & sub2["adsh"].isin(df["adsh"].unique())
    ]
    # for few recently uploaded submissions there will figures in companyfacts.zip
    # but submission information in submissions.zip.
    missing_adsh = np.setdiff1d(df["adsh"], np.union1d(sub["adsh"], sub2["adsh"]))
    sub3 = read_missing_submissions(missing_adsh, fetcher)
    if sub3 is not None:
        sub2 = pd.concat((sub2, sub3))

    sub2 = update_version_info(sub2, fetcher=fetcher, logger=logger)
    sub = sub[~sub["adsh"].isin(sub2["adsh"])].drop(columns="file")
    logger.info("Version information loaded")

    # Combine the submissions data, set amendment flags, and merge with the ticker data
    sub = (
        pd.concat((sub, sub2), ignore_index=True)
        .pipe(set_amended_flag)
        .merge(tickers, how="inner", on="cik")
    )

    return df, sub
	
	
def repair_version(sub):
    # Step 1: Create a version map for amendment reports.
    # This map links amendment reports to the versions of the reports 
    # they are amending.
    version_map = (
        sub[["amendment_adsh", "version"]]
        .query("version > 0 and amendment_adsh > 0")
        .set_index("amendment_adsh")
        .to_dict(orient="dict")["version"]
    )
    
    # Step 2: Assign the version from the map to reports with a missing version 
    # If a report has version 0, look up its amendment_adsh in the version_map 
    # and assign the corresponding version. If no match is found, assign version 0.
    sub.loc[sub["version"] == 0, "version"] = (
        sub.loc[sub["version"] == 0, "adsh"].map(version_map).fillna(0).astype(int)
    )
    
    # Step 3: For reports that still have a missing version (version is NaN),
    # carry forward the version from the previous report for the same CIK, 
    # based on acceptance date.
    sub.sort_values(by=["cik", "accepted"], inplace=True)
    sub["version"] = sub["version"].replace(0, np.nan)
    sub["version"] = sub.groupby("cik", as_index=False)["version"].ffill()  
    
    # Step 4: For any remaining reports with a missing version,
    # carry forward the version from the next report for the same CIK, 
    # in reverse order of acceptance date.
    sub.sort_values(by=["cik", "accepted"], ascending=[True, False], inplace=True)
    sub["version"] = sub.groupby("cik", as_index=False)["version"].ffill().astype(int)
    
    # Return the DataFrame with the repaired version information.
    return sub
	
	
def get_submission_attrib(adsh, cik, fetcher):
    """
    Retrieves the US-GAAP version from the FilingSummary.xml file on the SEC EDGAR database
    for a given submission identified by its ADSH and CIK.

    Parameters:
    - adsh: Accession Number (ADSH) of the submission.
    - cik: Central Index Key (CIK) of the filing entity.
    - fetcher: URLFetcher class.

    Returns:
    - file: The primary document file name (10-Q, 10-K, or their amendments).
    - ns_list: A list of namespaces related to US-GAAP or DEI taxonomies.
    """
    # Construct the URL for the FilingSummary.xml file on the SEC EDGAR site
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik}/{adsh:018d}/FilingSummary.xml"
    )
    # Send the request 
    resp = fetcher.fetch(url, ignore_exceptions=True)
    if resp is None:
        return None, None

    # Parse the XML content from the response
    root = ElementTree.parse(resp).getroot()
    resp.close()

    # Find the primary document file (e.g., 10-Q, 10-K) from the XML
    file = None
    for f in root.findall("InputFiles/File"):
        if "doctype" in f.attrib and f.attrib["doctype"] in [
            "10-Q",
            "10-Q/A",
            "10-K",
            "10-K/A",
        ]:
            file = f.text
            break

    # Extract the list of relevant namespaces (US-GAAP, DEI)
    ns_list = []
    for ns in root.findall("BaseTaxonomies/BaseTaxonomy"):
        if "/dei/" in ns.text or "/us-gaap/" in ns.text:
            ns_list += [ns.text]

    return file, ns_list


def get_context(root, cik):
    """
    Extracts the context information from an XML submission.
    The context includes entity identifiers and the start/end dates for financial figures.

    Parameters:
    - root: The root element of the parsed XML document.
    - cik: Central Index Key (CIK) of the filing entity.

    Returns:
    - A pandas DataFrame containing context references, entity IDs, and start/end dates
      specific to the given CIK.
    """
    xbrl_ns = {"xbrl": "http://www.xbrl.org/2003/instance"}
    context = root.findall("xbrl:context", namespaces=xbrl_ns)
    cnt_list = []

    for c in context:
        cnt = c.attrib["id"]
        id = int(
            c.find("xbrl:entity", namespaces=xbrl_ns)
            .find("xbrl:identifier", namespaces=xbrl_ns)
            .text
        )
        try:
            start_date = (
                c.find("xbrl:period", namespaces=xbrl_ns)
                .find("xbrl:startDate", namespaces=xbrl_ns)
                .text
            )
            end_date = (
                c.find("xbrl:period", namespaces=xbrl_ns)
                .find("xbrl:endDate", namespaces=xbrl_ns)
                .text
            )
        except:
            end_date = (
                c.find("xbrl:period", namespaces=xbrl_ns)
                .find("xbrl:instant", namespaces=xbrl_ns)
                .text
            )
            start_date = end_date

        cnt_list += [
            {
                "contextRef": cnt,
                "entity": id,
                "start": start_date,
                "end": end_date,
            }
        ]

    return pd.DataFrame(cnt_list).query("entity==@cik")


def get_units(root):
    """
    Extracts unit information from an XML submission, focusing on USD and shares.

    Parameters:
    - root: The root element of the parsed XML document.

    Returns:
    - A list of unit IDs corresponding to monetary (USD) and shares measurements.
    """
    xbrl_ns = {"xbrl": "http://www.xbrl.org/2003/instance"}
    units = root.findall("xbrl:unit", namespaces=xbrl_ns)
    id_list = []

    for u in units:
        id = u.attrib["id"]
        try:
            measure = u.find("xbrl:measure", namespaces=xbrl_ns).text
        except:
            continue

        # Filter to include only USD or shares-related units
        if (
            measure in ["shares", "USD"]
            or measure.endswith(":USD")
            or measure.endswith(":shares")
        ):
            id_list += [id]

    return id_list


def get_submission(root, cik, ns_list):
    """
    Extracts financial figures from an XML submission
    based on the relevant namespaces.

    Parameters:
    - root: The root element of the parsed XML document.
    - cik: Central Index Key (CIK) of the filing entity.
    - ns_list: List of namespaces relevant to US-GAAP or DEI taxonomies.

    Returns:
    - A pandas DataFrame containing the extracted financial figures,
      with associated tags, dates, and values.
    """
    df_list = []

    for ns in ns_list:
        # Extract relevant data from each namespace
        df = pd.DataFrame(
            [
                r.attrib | {"tag": r.tag} | {"value": r.text}
                for r in root.findall("ns:*", namespaces={"ns": ns})
            ]
        )
        if len(df) == 0:
            continue

        # Clean up the tag names by removing the namespace prefix
        df["tag"] = df["tag"].replace("{" + ns + "}", "", regex=True)
        df_list += [df]

    df = pd.concat(df_list)
    df["value"] = df["value"].fillna(0.0)

    # Filter the data to include only relevant units (USD or shares)
    if "unitRef" in df.columns:
        df = df[df["unitRef"].isin(get_units(root))]

    # Merge with context information and return the relevant columns
    return df.merge(get_context(root, cik))[["tag", "start", "end", "value"]].copy()


def read_missing_figures(sub, tag_list, fetcher, logger):
    """
    Reads financial figures for submissions that are missing data.

    Parameters:
    - sub: DataFrame containing submissions with missing figures.
    - tag_list: List of tags (financial metrics) to extract.
    - fetcher: URLFetcher class
    - logger: Logger object for logging information and warnings.

    Returns:
    - A DataFrame containing the extracted financial figures
      for the missing submissions, filtered by the specified tags.
    """
    df_list = []

    for index, row in sub.iterrows():

        cik, adsh = row["cik"], row["adsh"]
        file, ns_list = get_submission_attrib(adsh, cik, fetcher)

        if file is None:
            logger.warning(f"No main submission file found for submission {adsh}")
            continue

        # Adjust the file extension if necessary
        if not file.endswith(".xml"):
            file = "_htm".join(file.rsplit(".htm", 1)) + ".xml"

        # Construct the URL for the primary document file
        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{row['cik']}/{row['adsh']:018d}/{file}"
        )
        resp = fetcher.fetch(url, ignore_exceptions=True)
        if resp is not None:
            # Parse the XML content and extract the relevant figures
            root = ElementTree.parse(resp).getroot()
            resp.close()
            df = get_submission(root, cik, ns_list).assign(adsh=adsh)
            df_list += [df]

    if len(df_list) == 0:
        return None

    # Combine the results into a single DataFrame
    df = pd.concat(df_list)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Filter out NaN values and keep only the specified tags
    df = df[~df["value"].isna() & df["tag"].isin(tag_list)]
    df["start"] = df["start"].astype("datetime64[s]")
    df["end"] = df["end"].astype("datetime64[s]")
    df["tag"] = pd.Categorical(df["tag"], categories=tag_list)

    return df


def read_missing_figures_2(fetcher, logger, df, sub):
    """
    Retrieves and combines additional financial figures
    that were missing from the initial DataFrame.

    Parameters:
    - df: The initial DataFrame containing financial figures.
    - sub: DataFrame containing submissions that potentially
           have missing figures.

    Returns:
    - A concatenated DataFrame with the missing figures added,
      or the original DataFrame if no additional figures were found.
    """
    tag_list = read_tags(fetcher)

    # Identify submissions without figures in the initial DataFrame
    sub_no_figures = sub[~sub["adsh"].isin(df["adsh"])]
    logger.info(f"There are {len(sub_no_figures)} reports without figures.")
    logger.info("Loading missing figures.")

    # Retrieve the missing figures
    df2 = read_missing_figures(sub_no_figures, tag_list, fetcher, logger)
    logger.info("Additional figures loaded.")

    # Combine the initial and additional DataFrames, if any new figures were found
    return pd.concat((df, df2)) if (df2 is not None) else df
	
	
def extract_submissions_and_facts(logger, debug_mode=False):
	fetcher = URLFetcher(logger)
	# finally run the extraction
	df, sub = extract_submissions_and_facts_internal(fetcher, logger, debug_mode)
	sub = repair_version(sub)
	df = read_missing_figures_2(fetcher, logger, df, sub).reset_index(drop=True)
	return df, sub
	
	
def check_subs(logger, sub):
    # Check that size of set is big enough
    assert len(sub) > 200000, "Too few lines in submissions"
    # Check number of columns
    assert len(sub.columns) == 10, "Sumbissions should have exactly 10 columns"
    # Create a dictionary representing the DataFrame structure
    df_structure = {
        "adsh": np.int64,
        "cik": np.int64,
        "sic": np.int64,
        "form": object,
        "period": np.dtype("datetime64[s]"),
        "accepted": np.dtype("datetime64[s]"),
        "version": np.int64,
        "amendment_adsh": np.int64,
        "is_amended": bool,
        "ticker": pd.CategoricalDtype,
    }
    for c in sub.columns:
        if c != "ticker":
            assert sub[c].dtype == df_structure[c], f"Wrong type for column {c}"
    assert sub["adsh"].max() < 1e16, "Some ADSH values are too high"
    assert sub["sic"].max() < 10000, "Some SIC values are too high"
    assert sub["period"].min().year > 2003, "Some periods are too early "
    assert sub["period"].max().year < 2030, "Some periods are too late "
    assert sub["accepted"].min().year > 2003, "Some acceptance dates are too early "
    assert sub["accepted"].max().year < 2030, "Some acceptance dates are too late "
    assert np.isin(
        np.setdiff1d(sub["amendment_adsh"], [0]), sub["adsh"].unique()
    ).all(), "Some amendment_adsh are not in ADSH list"
    assert (
        len(sub.query("amendment_adsh>0 and not is_amended")) == 0
    ), "amendment_adsh!=0, but is_amended flag is False "
    assert (
        len(sub.query("amendment_adsh==0 and is_amended")) == 0
    ), "amendment_adsh=0, but is_amended flag is True "
    existing_versions = [0, 2008, 2009] + list(range(2011, 2025))
    assert (
        ~sub["version"].isin(existing_versions)
    ).sum() == 0, "there are unknown version numbers"
    assert (
        ~sub["form"].isin(["10-Q", "10-K", "10-Q/A", "10-K/A"])
    ).sum() == 0, "there are unknown forms"
    subextr = sub[
        (sub.adsh == 119312522137021) | (sub.adsh == 110465923054237)
    ].reset_index(drop=True)
    subextr["ticker"] = subextr["ticker"].astype(str)
    data = {
        "adsh": [119312522137021, 110465923054237],
        "cik": [1878897, 1902700],
        "sic": [6531, 2833],
        "form": ["10-K/A", "10-K/A"],
        "period": [pd.to_datetime("2021-12-31"), pd.to_datetime("2022-12-31")],
        "accepted": [
            pd.to_datetime("2022-05-02 17:18:57"),
            pd.to_datetime("2023-05-01 15:10:29"),
        ],
        "version": [2021, 2021],
        "amendment_adsh": [0, 141057823001429],
        "is_amended": [False, True],
        "ticker": pd.Categorical(["doug", "pgff"]),
    }
    assert len(subextr.compare(pd.DataFrame(data))) == 0, "Selective check failed"
    # For 9 reports the reporting period is after submission date
    # We will ignore these 9 reports, but raise error for other reports
    # Example, for adsh=147124212001044 or 0001471242-12-001044
    # https://data.sec.gov/submissions/CIK0001508381.json
    # submission #502, reportDate = "2012-12-31", acceptanceDateTime = "2012-08-07T15:30:10.000Z""
    # and in file
    # https://www.sec.gov/files/dera/data/financial-statement-data-sets/2012q3.zip
    # accepted = "2012-08-07 15:30:00.0"
    future_periods = [
        88616314000119,
        72174814000106,
        109690621001302,
        156276213000125,
        147124212001044,
        149315224018671,
        162528517000005,
        159991622000122,
        183568124000069,
    ]
    assert (
        len(sub[(sub["accepted"] < sub["period"]) & ~sub["adsh"].isin(future_periods)])
        == 0
    ), "There are reports with future periods"
    assert len(sub.query("version==0")) == 0, "There are reports without versions"
    logger.info("Submissions checks OK")
	
	
def check_figures(logger, df, sub):
    # Check that size of set is big enough
    assert len(df) > 50_000_000, "Too few lines in facts"
    # Check number of columns
    assert len(df.columns) == 5, "Sumbissions should have exactly 5 columns"
    df_structure = {
        "adsh": np.int64,
        "tag": "category",
        "start": np.dtype("datetime64[s]"),
        "end": np.dtype("datetime64[s]"),
        "value": np.float64,
    }
    for c in df.columns:
        assert df[c].dtype == df_structure[c], f"Wrong type for column {c}"
    assert (df["start"] <= df["end"]).all(), "Start date before end date"
    figsample = df[
        (df.adsh == 156459021039151)
        & df["tag"].isin(
            [
                "AdvertisingExpense",
                "AllocatedShareBasedCompensationExpense",
                "AmortizationOfIntangibleAssets",
            ]
        )
    ].reset_index(drop=True)
    data = {
        "adsh": [156459021039151] * 9,
        "tag": ["AdvertisingExpense"] * 3
        + ["AllocatedShareBasedCompensationExpense"] * 3
        + ["AmortizationOfIntangibleAssets"] * 3,
        "start": pd.to_datetime(
            [
                "2018-07-01",
                "2019-07-01",
                "2020-07-01",
                "2018-07-01",
                "2019-07-01",
                "2020-07-01",
                "2018-07-01",
                "2019-07-01",
                "2020-07-01",
            ]
        ),
        "end": pd.to_datetime(
            [
                "2019-06-30",
                "2020-06-30",
                "2021-06-30",
                "2019-06-30",
                "2020-06-30",
                "2021-06-30",
                "2019-06-30",
                "2020-06-30",
                "2021-06-30",
            ]
        ),
        "value": [1.6e9, 1.6e9, 1.5e9, 4.652e9, 5.289e9, 6.118e9, 1.9e9, 1.6e9, 1.6e9],
    }
    data = pd.DataFrame(data)
    data["tag"] = pd.Categorical(data["tag"], categories=df["tag"].cat.categories)
    assert len(figsample.compare(data)) == 0, "Selective check failed"
    # TODO - there are figures not assigned to a record in sub. NEED TO CHECK THEM
    assert (
        len(df[(~df["adsh"].isin(sub["adsh"]))]["adsh"].unique()) < 55
    ), "There are figures not assigned to a report"
    # There are few reports without figures
    # For example, the report: 0001213900-24-032308
    # Location: https://www.sec.gov/Archives/edgar/data/1881551/000121390024032308/0001213900-24-032308-index.html - has facts
    # https://www.sec.gov/Archives/edgar/data/1881551/000121390024032308
    # not present in database
    # in facts file: https://data.sec.gov/api/xbrl/companyfacts/CIK0001881551.json - no facts present, latest fact on 2023-09-30, no data for 2023-12-31
    # probably the problem is in ticker, but it is not possible to say it for sure
    # We will ignore these reports, but check all others
    sub_no_data = [
       114420411065305, 114420411053088, 104746909007400, 110852420000029,
       128703223000355, 155837022007084, 119312511051841, 138713117005367,
       138713117003911, 138713117002568, 119312512266785, 107878220001010,
       144530513002979, 158798723000073, 119312519209634, 160971115000009,
       165365316000008]
    assert (
        len(sub[~sub["adsh"].isin(df["adsh"]) & ~sub["adsh"].isin(sub_no_data)]) == 0
    ), "There are reports without figures"
    logger.info("Figures checks OK")
	
	
def check_submissions_and_facts(logger, df, sub):
	check_subs(logger, sub)
	check_figures(logger, df, sub)
