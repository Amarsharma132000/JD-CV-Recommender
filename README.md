# JD-CV-Recommender
AI-Powered Recruitment Platform: End-to-end automated candidate discovery system using agentic AI principles. Processes job descriptions, scrapes LinkedIn via Google X-Ray search, and ranks candidates with multi-modal AI scoring (BERT embeddings, BM25, vector similarity). Reduces manual screening time through intelligent automation.


# JD-CV Recommendation System

## Project Goal

The JD-CV Recommendation System aims to automate the process of matching job descriptions (JDs) with the most relevant candidate profiles (CVs/online profiles). By leveraging a multi-stage approach, the system first identifies a pool of potential candidates and then ranks them based on their relevance to a given JD, providing a more intelligent recommendation than simple keyword matching.

## Project Approach

The system follows a three-step process:

1.  **JD Parsing**: Extracts key information from a job description, including a bag of keywords, job role, required experience, and location.
2.  **Candidate Pool Generation**: Uses Google X-Ray search (via SerpApi) with pagination to find a preliminary list of potential candidates by collecting public LinkedIn profile URLs based on search criteria derived from the JD.
3.  **JD-CV Recommender**: Takes the collected LinkedIn profile URLs and the job role as input. It utilizes the SignalHire API to enrich the profile data and then employs a hybrid ranking mechanism (`JobRoleMatcher`) combining:
    *   **Semantic Similarity**: Using Sentence Embeddings (SentenceTransformer) and Annoy for efficient similarity search between the job role and candidate profiles.
    *   **Keyword Matching**: Using the BM25 algorithm to score the relevance of keywords from the JD within candidate profiles.
    *   **Weighted Scoring**: Calculates a final AI Score as a weighted sum of semantic similarity, BM25, and direct keyword match scores to produce a ranked list of candidates.

## Tech Stack

*   **Python**: Core programming language.
*   **`google-search-results` (SerpApi)**: For Google X-Ray searches to find LinkedIn URLs.
*   **`pyngrok`**: To create a secure tunnel for SignalHire API callbacks in Colab.
*   **`requests`**: For making HTTP requests to external APIs.
*   **`pandas`**: For data handling and manipulation.
*   **`numpy`**: For numerical operations.
*   **`scikit-learn`**: General machine learning utilities.
*   **`annoy`**: For efficient approximate nearest neighbor search on embeddings.
*   **`rank-bm25`**: Implementation of the BM25 algorithm for keyword scoring.
*   **`nltk`**: For natural language processing (tokenization, stop words, lemmatization).
*   **`transformers` & `torch`**: Underlying libraries for the SentenceTransformer model.
*   **`sentence-transformers`**: For generating sentence embeddings.
*   **`scipy`**: For cosine similarity calculations.
*   **SignalHire API**: External service for enriching LinkedIn profile data.

## Deliverable Results

*   A ranked list of candidates (pandas DataFrame) based on their calculated AI Score.
*   Enriched candidate profile data (position, skills, summary) retrieved via SignalHire.
*   Individual scores (role similarity, BM25, keyword) providing insight into the ranking factors.
*   A scalable process for generating a candidate pool from Google search results.
*   Downloadable JSON files containing the retrieved profile data.

## Setup and Usage

1.  **Clone the repository:** Clone this repository to your local machine or open it directly in Google Colab.
2.  **Open in Google Colab:** It is recommended to run this project in Google Colab due to the dependencies and the ngrok setup for the callback server.
3.  **Secure API Keys:**
    *   Obtain API keys for SerpApi and SignalHire.
    *   In Google Colab, go to the 'Secrets' tab (key icon) in the left sidebar.
    *   Add a new secret named `API_KEY` for your SerpApi key.
    *   Add another secret named `SIGNALHIRE_API_KEY` for your SignalHire API key.
    *   Ensure 'Notebook access' is enabled for these secrets.
.
4.  **Install Dependencies:** Run the first code cell to install all required libraries (`google-search-results`, `pyngrok`, `requests`, `pandas`, `numpy`, `scikit-learn`, `annoy`, `rank-bm25`, `nltk`, `transformers`, `torch`, `sentence-transformers`) and download NLTK data.
5.  **Set Ngrok Authtoken:** Run the code cell containing `ngrok.set_auth_token("YOUR_AUTH_TOKEN")` and replace `"YOUR_AUTH_TOKEN"` with your actual ngrok authtoken obtained from your ngrok dashboard.
6.  **Run Cells Sequentially:** Execute the code cells in the notebook from top to bottom.
    *   The SerpApi cell will generate a list of LinkedIn URLs based on the defined query and save them to `linkedin_urls.txt`.
    *   The SignalHire cell will start a callback server, set up an ngrok tunnel, read URLs from `linkedin_urls.txt`, send them to the SignalHire API for enrichment, and save the results to `profiles_bulk.json` and `profiles_intermediate_bulk.json`.
    *   The data loading cell will load the enriched data into a pandas DataFrame `df`.
    *   The JobRoleMatcher cell will initialize the `JobRoleMatcher` class with the loaded data, defined job roles, and keywords, and then run the matching process, producing the `result_df`.
7.  **Review Results:** The notebook will display the ranked candidates with their AI Score. You can further analyze the `result_df` to see the individual scoring components.
8.  **Download Outputs:** The `profiles_bulk.json` and `profiles_intermediate_bulk.json` files containing the enriched profile data will be available for download from the Colab environment.

## Notes

*   The effectiveness of the candidate pool generation depends on the Google X-Ray search query used.
*   The performance and cost of the SignalHire API calls depend on the number of URLs processed and your SignalHire subscription.
*   The weights (`a`, `b`, `c`) in the `JobRoleMatcher` can be adjusted to tune the importance of semantic similarity, BM25, and keyword matching.
*   The `n_gram` parameter in `JobRoleMatcher` controls the range of N-grams used for keyword matching.
"""
