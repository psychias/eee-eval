#!/usr/bin/env python3
"""
Fetch real papers from arXiv related to LLM evaluation for your paper coverage.
This script queries the arXiv API for papers matching your research area.
"""

import urllib.request
import xml.etree.ElementTree as ET
import time
from datetime import datetime
from typing import List, Dict
from urllib.parse import quote

def search_arxiv(query: str, max_results: int = 25,
                 years: List[int] = [2026, 2025, 2024]) -> List[Dict]:
    """
    Search arXiv using urllib and XML parsing.

    Args:
        query: Search query string
        max_results: Maximum papers to return
        years: Years to include (as filter hint)

    Returns:
        List of paper dicts with title, authors, url, date
    """
    # arXiv API endpoint
    base_url = "http://export.arxiv.org/api/query?"

    # Build query - cs.CL = Computation & Language
    query_encoded = quote(query)
    full_query = f"search_query=cat:cs.CL+AND+({query_encoded})&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"

    url = base_url + full_query
    print(f"Searching: {query}")

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        papers = []

        # Define namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        for entry in root.findall('atom:entry', ns):
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text if title_elem is not None else 'N/A'

            # Get authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text)

            # Get ID and construct URL
            id_elem = entry.find('atom:id', ns)
            arxiv_id = id_elem.text.split('/abs/')[-1] if id_elem is not None else 'N/A'
            url_paper = f"https://arxiv.org/abs/{arxiv_id}"

            # Get published date
            published_elem = entry.find('atom:published', ns)
            published = published_elem.text if published_elem is not None else 'N/A'

            # Get summary
            summary_elem = entry.find('atom:summary', ns)
            summary = summary_elem.text if summary_elem is not None else ''
            summary = ' '.join(summary.split())  # Clean whitespace
            summary = summary[:200] + '...' if len(summary) > 200 else summary

            paper = {
                'title': title,
                'authors': ', '.join(authors),
                'url': url_paper,
                'arxiv_id': arxiv_id,
                'published': published,
                'summary': summary
            }
            papers.append(paper)

        return papers
    except Exception as e:
        print(f"Error searching arXiv: {e}")
        return []

def format_bibtex(paper: Dict) -> str:
    """Convert paper to BibTeX format."""
    arxiv_id = paper['arxiv_id']
    year = paper['published'][:4]

    # Create short name from first author + year + first word
    first_author = paper['authors'].split(',')[0].split()[-1].lower()
    first_word = paper['title'].split()[0].lower().replace(':', '')
    short_name = f"{first_author}-{year}-{first_word}"

    bibtex = f"""@article{{{short_name},
  author    = {{{paper['authors']}}},
  title     = {{{paper['title']}}},
  journal   = {{arXiv preprint arXiv:{arxiv_id}}},
  year      = {{{year}}},
  url       = {{https://arxiv.org/abs/{arxiv_id}}},
}}"""
    return bibtex

def main():
    # Search queries covering your coverage gaps
    queries = [
        "prompt template LLM evaluation",
        "temperature sampling LLM benchmark",
        "few-shot learning instability",
        "log-likelihood vs generation scoring",
        "evaluation harness comparison",
        "benchmark contamination detection",
        "instruction tuning effects benchmark",
        "leaderboard reproducibility",
        "LLM evaluator calibration",
        "benchmark suite design evaluation",
        "score divergence LLM models",
        "n-shot prompt sensitivity",
        "chain-of-thought evaluation variance",
        "harness implementation differences",
        "evaluation specification",
    ]

    all_papers = []

    for query in queries:
        papers = search_arxiv(query, max_results=10)
        all_papers.extend(papers)
        time.sleep(1)  # Be respectful to arXiv API

        # Print found papers
        for i, paper in enumerate(papers[:5], 1):  # Show first 5
            print(f"{i}. {paper['title']}")
            print(f"   Authors: {paper['authors']}")
            print(f"   URL: {paper['url']}")
            print(f"   Date: {paper['published'][:10]}\n")

    # Deduplicate by URL
    unique_urls = set()
    unique_papers = []
    for paper in all_papers:
        if paper['url'] not in unique_urls:
            unique_urls.add(paper['url'])
            unique_papers.append(paper)

    # Sort by date (newest first)
    unique_papers.sort(key=lambda x: x['published'], reverse=True)

    # Take top 25
    top_papers = unique_papers[:25]

    print(f"\n{'='*70}")
    print(f"FOUND {len(top_papers)} UNIQUE PAPERS - FORMATTED FOR YOUR BIB FILE")
    print(f"{'='*70}\n")

    # Create output: plain text URLs
    output_urls = []
    output_bibtex = []

    for i, paper in enumerate(top_papers, 1):
        output_urls.append(f"{i}. {paper['title']}")
        output_urls.append(f"   Authors: {paper['authors']}")
        output_urls.append(f"   {paper['url']}")
        output_urls.append("")

        output_bibtex.append(format_bibtex(paper))
        output_bibtex.append("")

    # Write files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(f'arxiv_papers_{timestamp}.txt', 'w') as f:
        f.write("PAPERS TO ADD TO YOUR COVERAGE (2024-2026)\n")
        f.write("="*70 + "\n\n")
        f.write('\n'.join(output_urls))

    with open(f'arxiv_papers_{timestamp}.bib', 'w') as f:
        f.write("% Auto-generated from arXiv search\n")
        f.write("% Copy these entries into custom.bib\n\n")
        f.write('\n'.join(output_bibtex))

    print("Saved to:")
    print(f"  - arxiv_papers_{timestamp}.txt (URLs)")
    print(f"  - arxiv_papers_{timestamp}.bib (BibTeX entries)")
    print("\nCopy the BibTeX entries to custom.bib and cite in your paper.")

if __name__ == "__main__":
    main()
