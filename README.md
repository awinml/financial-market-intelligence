# Financial Dashboard for Market Intelligence

- Built a end-to-end financial dashboard that collects and consolidates all of a business's critical observations in one place using the information obtained from the annual 10-K SEC Filings of 12 companies.

- Collected text data from 10-K filings from SEC EDGAR using the [SEC ExtractorAPI](https://sec-api.io/). 

- The filings of 12 companies spanning 5 sectors were collected for the duration of 2017 to 2021. Each filing had over 34,000 words.

- The data was cleaned and transformed for Sentiment Analysis and Summarization. The data was manually labelled for both the tasks.

- The **RoBERTa, FinBERT and DistilBERT** models were fine-tuned for sentiment analysis. The best results were obtained using the fine-tuned DistilBERT model. It achieved an **Accuracy of 91.11% and an ROC-AUC Score of 0.972.**

- The **T5, DistilPEGASUS and DistilBART** models were fine-tuned for summarization. The best results were obtained using the fine-tuned DistilBART model. It achieved an **ROUGE-L Score of 67.7%.**

- RAKE NLTK was used to identify important keywords from the generated summaries.

- The Financial Dashboard was deployed as a web-app using Streamlit. It contains:
    - **Insights and summaries** for different sections from annual corporate filings.
    - Identification of **important topics** mentioned in the report.
    - **Sentiment-based score** that measures the company's performance over a certain time period.

The app can be viewed here: [Financial Dashboard](https://awinml-financial-market-intelligence-app-q6lj0g.streamlit.app/)

<!---
## Why do we need a consolidated Financial Dashboard?
In the current data driven world, it is essential to have access to the right information for impactful decision making. All publicly listed companies have to file annual reports to the government. These consolidated statements allow investors, financial analysts, business owners and other interested parties to get a complete overview of the company. Companies all over the world make key financial decisions based on annually released public filings. 

These corporate filings are rife with complicated legal and financial jargon and make it practically impossible for a layman to understand. In most cases these documents have to be manually read and decoded by people with expert financial and legal understanding. The goal of this project is to develop a tool that automates this tedious procedure and makes it easier to acquire crucial financial information.

The results of the modelling can be found here:

-->


