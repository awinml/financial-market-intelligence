# https://sec-api.io/
from sec_api import ExtractorApi  
import pandas as pd  
import re



def return_company_name(url):
    """
    Fetch Company's Name from URL string.

    Parameters
    ----------
    url : str
        URL of report.

    Returns
    -------
    str
        Company Name.
    """
    company_name = []

    for i in url[::-1]:
        if i == '/':
        break
        
        company_name.append(i)

    return ''.join(company_name[-4:])[::-1]



def extractor_filing(extractorApi, filing_url, section):
    """
    extractor_filing 
        Extract required sections from 10-K filing and return a Dataframe of text extracted.

    Parameters
    ----------
    extractorApi : 
        API Key Instance
    filing_url : str
        URL of report.
    section : str
        Section name

    Returns
    -------
    Pandas DataFrame
        Extracted Text
    """

    rows = []

    for url in filing_url:

        dict_ = {}

        columns = ["Company", "Reporting_Date"]

        elements = [return_company_name(url), url[-12:-4]]

        for sec in section:

        extracted_section = extractorApi.get_section(url, sec, "text")
        cleaned_section = re.sub(r"\n|&#[0-9]+;", "", extracted_section)

        columns.append(sec)
        elements.append(cleaned_section)
        
        rows.append(elements)

    return pd.DataFrame(rows, columns=columns)

extractorApi = ExtractorApi("fcb2dcbdd4fead6b697c30ec7f025e0a00cb65eb25318217be3396d0950e49e3")

# URLs of 10-K filings to extract

filing_url = [
    "https://www.sec.gov/Archives/edgar/data/320193/000032019321000105/aapl-20210925.htm",
    "https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/aapl-20200926.htm",
    "https://www.sec.gov/Archives/edgar/data/320193/000032019319000119/a10-k20199282019.htm",
    "https://www.sec.gov/Archives/edgar/data/320193/000032019318000145/a10-k20189292018.htm",
    "https://www.sec.gov/Archives/edgar/data/320193/000032019317000070/a10-k20179302017.htm",
    "https://www.sec.gov/Archives/edgar/data/320193/000162828016020309/a201610-k9242016.htm",
    "https://www.sec.gov/Archives/edgar/data/320193/000119312515356351/d17062d10k.htm",
    "https://www.sec.gov/Archives/edgar/data/320193/000119312514383437/d783162d10k.htm",
    "https://www.sec.gov/Archives/edgar/data/320193/000119312513416534/d590790d10k.htm",
    "https://www.sec.gov/Archives/edgar/data/320193/000119312512444068/d411355d10k.htm"
]

# Sections list to be extracted

section = ["1", "1A", "1B", "2", "3", "5", "7", "7A", "8", "9A"]

results = extractor_filing(extractorApi, filing_url, section)
results.to_csv('tesla.csv')