# Jane Austen Revisited: A Computational Approach to Authorship Attribution

## Executive Summary

The purpose of this project is to explore authorship attribution through a computational lens, comparing the performance of a Large Language Model (LLM) with various Natural Language Processing (NLP) methods. Authorship attribution is pivotal in verifying text authenticity, detecting plagiarism, and understanding authorial style. By analyzing the works of Jane Austen alongside selected fan works based on her books, the project aims to train models to distinguish between Austen's original writings and imitative fan works.

## Motivation

Authorship attribution has historical significance, evidenced by debates surrounding the works of Shakespeare and the Federalist Papers. Many works throughout history were published anonymously, and some women authors, like Jane Austen, opted for anonymity due to societal pressures. Accurate authorship attribution is crucial in cases of plagiarism, copyright disputes, and preserving literary legacies.

## Data Question

Can machine learning models accurately distinguish between Jane Austen's original works and fan-authored works imitating her style? Which linguistic features are most influential in these models? The project aims to investigate linguistic differences, commonality of proper nouns, direct copying, and sentiment variations between Austen and modern fan authors.

## Minimum Viable Product

The project will train and evaluate at least two types of models, presenting the analysis results in a slide deck. Additional features and model types will be explored, and there's a plan to develop an interactive website or app if time permits.

## Schedule (through 6/13/2024)

1. Get the Data (5/18/2024)
2. Clean & Explore the Data (6/4/2024)
3. Create Presentation (6/6/2024)
4. Internal Demos (6/8/2024)
5. Demo Day (6/13/2024)

## Data Sources

Jane Austen's novels will be sourced from Project Gutenberg in plain text format.

[Project Gutenberg](https://www.gutenberg.org)

Fan fiction will be collected from Archive of Our Own, filtered by tags to ensure relevance and exclude fantastical elements or other fandoms.

[Archive of Our Own](https://archiveofourown.org)

## Known Issues and Challenges

Data cleaning is necessary, especially to remove metadata and HTML tags. Fan works may utilize established characters and locations from Austen's novels, potentially inflating model performance. To address this, modifications will be made to character and location names, and models will be trained on both original and modified texts for comparison.

---------------------------------------------------------------

## Data Acquisition and Cleaning

### Novels of Jane Austen from Project Gutenberg
Project Gutenberg has a catalog file available for download. This file was searched to obtain metadata for the appropriate texts. Austen's novels were collected by Project Gutenberg, so some books had several versions available. When multiple English versions of a particular novel were available, the most recent file was chosen. The 'gutenbergpy' package was used with the text id to retrieve the files and strip the header and footer information. Each book was written to a text file with the text id as the file name. The metadata was written to a csv file.

### Fan Fiction Works based on Jane Austen
Fan fiction works published on the website 'Archive of Our Own' (AO3) were collected for this project. Several search parameters were used to hone in on works most suitable for this project including eliminating works under 5000 words and incomplete works. In addition, the available tags were used to select works that most closely resembled the original setting of Austen's novels, so stories with modern settings or fantastical elements were also excluded. 350 stories met all the criteria and the metadata was collected using the 'ao3downloader' script. The work id was extracted from the data and used in a webscraping script; each story was extracted from the html formatting and saved as a text file with the work id as the file name. The metadata was saved as a csv file.