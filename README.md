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

