import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin, urlparse
import re
import html

BASE_URL = "https://support.tngdigital.com.my/hc/en-my"
API_BASE_URL = "https://support.tngdigital.com.my/api/v2/help_center/en-my"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def get_soup(url):
    """Fetch and parse a URL"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.content, "html.parser")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def extract_text(element):
    """Extract clean text from an element"""
    if element is None:
        return ""
    # Remove script and style elements
    for script in element(["script", "style"]):
        script.decompose()
    text = element.get_text(separator=" ", strip=True)
    # Clean up whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_categories():
    """Get all categories from the main page"""
    soup = get_soup(BASE_URL)
    if not soup:
        return []

    categories = []
    # Look for category links - Zendesk Help Center typically has categories in navigation
    # Try multiple selectors that might be used
    category_links = soup.find_all("a", href=re.compile(r"/hc/en-my/categories/\d+"))

    for link in category_links:
        href = link.get("href")
        if href:
            full_url = urljoin(BASE_URL, href)
            name = extract_text(link)
            if name and full_url not in [c["url"] for c in categories]:
                categories.append({"name": name, "url": full_url})

    # Also check for section links if categories aren't found
    if not categories:
        section_links = soup.find_all("a", href=re.compile(r"/hc/en-my/sections/\d+"))
        for link in section_links:
            href = link.get("href")
            if href:
                full_url = urljoin(BASE_URL, href)
                name = extract_text(link)
                if name and full_url not in [c["url"] for c in categories]:
                    categories.append({"name": name, "url": full_url})

    return categories


def get_sections(category_url, category_name):
    """Get all sections within a category"""
    soup = get_soup(category_url)
    if not soup:
        return []

    sections = []
    # Look for section links
    section_links = soup.find_all("a", href=re.compile(r"/hc/en-my/sections/\d+"))

    for link in section_links:
        href = link.get("href")
        if href:
            full_url = urljoin(BASE_URL, href)
            name = extract_text(link)
            if name and full_url not in [s["url"] for s in sections]:
                sections.append(
                    {"name": name, "url": full_url, "category": category_name}
                )

    # If no sections found, the category itself might be a section
    if not sections:
        sections.append(
            {"name": category_name, "url": category_url, "category": category_name}
        )

    return sections


def get_articles_from_section(section_url, section_name, category_path):
    """Get all articles from a section"""
    articles = []
    page = 1

    while True:
        # Zendesk pagination
        if page == 1:
            url = section_url
        else:
            url = f"{section_url}?page={page}"

        soup = get_soup(url)
        if not soup:
            break

        # Find article links
        article_links = soup.find_all("a", href=re.compile(r"/hc/en-my/articles/\d+"))

        if not article_links:
            break

        for link in article_links:
            href = link.get("href")
            if href:
                article_url = urljoin(BASE_URL, href)
                # Skip if we've already processed this article
                if article_url not in [a["url"] for a in articles]:
                    articles.append(
                        {
                            "url": article_url,
                            "title": extract_text(link),
                            "section": section_name,
                            "category_path": category_path,
                        }
                    )

        # Check for next page
        next_page = soup.find("a", {"rel": "next"}) or soup.find(
            "a", string=re.compile(r"Next|next", re.I)
        )
        if not next_page:
            break

        page += 1
        time.sleep(0.5)  # Be polite with rate limiting

    return articles


def extract_article_content(article_url, title, category_path):
    """Extract question, answer, and categories from an article page"""
    soup = get_soup(article_url)
    if not soup:
        return None

    # Extract question (usually the title or h1)
    question = title
    h1 = soup.find("h1")
    if h1:
        question_text = extract_text(h1)
        if question_text:
            question = question_text

    # Extract answer (usually in article body)
    answer = ""
    article_body = soup.find(
        "div", class_=re.compile(r"article-body|articleBody|article-content|body", re.I)
    )
    if not article_body:
        # Try other common selectors
        article_body = (
            soup.find("div", {"data-test-id": "article-body"})
            or soup.find("section", class_=re.compile(r"article|content", re.I))
            or soup.find("article")
        )

    if article_body:
        answer = extract_text(article_body)

    # Build categories array from category_path
    categories = (
        category_path.copy() if isinstance(category_path, list) else [category_path]
    )

    # Try to find breadcrumbs for additional category info
    breadcrumbs = soup.find("nav", class_=re.compile(r"breadcrumb", re.I))
    if breadcrumbs:
        breadcrumb_items = breadcrumbs.find_all("a")
        for item in breadcrumb_items:
            text = extract_text(item)
            if text and text not in categories:
                categories.append(text)

    return {
        "question": question,
        "answer": answer,
        "categories": categories,
        "url": article_url,
    }


def get_section_hierarchy(section_id, sections_cache):
    """Build the full hierarchy path for a section by traversing parent sections
    Returns list from root parent section to current section (leaf)
    Example: [Parent Section 1, Parent Section 2, Current Section]
    """
    hierarchy = []
    current_section_id = section_id
    visited = set()  # Prevent infinite loops

    # First, collect all sections in the chain (from current to root)
    section_chain = []
    while current_section_id and current_section_id not in visited:
        visited.add(current_section_id)

        # Get section details from cache or API
        if current_section_id not in sections_cache:
            section_url = f"{API_BASE_URL}/sections/{current_section_id}.json"
            section_response = requests.get(section_url, headers=HEADERS, timeout=10)
            if section_response.status_code == 200:
                section_data = section_response.json()
                sections_cache[current_section_id] = section_data["section"]
            else:
                break

        section = sections_cache[current_section_id]
        section_name = section.get("name", "")

        # Add to chain (from current to root)
        if section_name:
            section_chain.append(section_name)

        # Move to parent section
        current_section_id = section.get("parent_section_id")
        if current_section_id:
            time.sleep(0.1)  # Small delay to be respectful
        else:
            break

    # Reverse to get from root to leaf (Category > Parent Section 1 > Parent Section 2 > ... > Current Section)
    hierarchy = list(reversed(section_chain))

    return hierarchy


def scrape_via_api():
    """Try to scrape using Zendesk Help Center API (public, no auth needed)"""
    print("Attempting to scrape via Zendesk API...")
    all_articles = []
    sections_cache = {}  # Cache section data to avoid repeated API calls

    try:
        # Get all categories
        print("Fetching categories from API...")
        categories_url = f"{API_BASE_URL}/categories.json"
        response = requests.get(categories_url, headers=HEADERS, timeout=10)

        if response.status_code != 200:
            print(
                f"API not accessible (status {response.status_code}), falling back to web scraping"
            )
            return None

        categories_data = response.json()
        categories = categories_data.get("categories", [])
        print(f"Found {len(categories)} categories via API")

        # Process each category
        for idx, category in enumerate(categories, 1):
            category_id = category["id"]
            category_name = category["name"]
            print(f"\nProcessing category {idx}/{len(categories)}: {category_name}")

            # Get all sections in this category (with pagination)
            all_sections = []
            sections_url = (
                f"{API_BASE_URL}/categories/{category_id}/sections.json?per_page=100"
            )

            while sections_url:
                sections_response = requests.get(
                    sections_url, headers=HEADERS, timeout=10
                )
                if sections_response.status_code == 200:
                    sections_data = sections_response.json()
                    sections = sections_data.get("sections", [])
                    all_sections.extend(sections)

                    # Cache all sections for hierarchy building
                    for section in sections:
                        sections_cache[section["id"]] = section

                    # Check for next page
                    sections_url = sections_data.get("next_page")
                    time.sleep(0.1)  # Small delay
                else:
                    break

            sections = all_sections
            print(f"  Found {len(sections)} sections")

            if not sections:
                # Category might directly contain articles
                sections = [{"id": category_id, "name": category_name}]

            # Process each section
            for section in sections:
                try:
                    section_id = section["id"]
                    section_name = section["name"]

                    # Handle encoding issues when printing
                    try:
                        print(f"    Processing section: {section_name}")
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        safe_name = section_name.encode(
                            "utf-8", errors="replace"
                        ).decode("utf-8", errors="replace")
                        print(f"    Processing section: {safe_name}")

                    # Build full category hierarchy: Category > Parent Section 1 > Parent Section 2 > ... > Current Section
                    category_path = [category_name]

                    # Get full section hierarchy (all parent sections from root to current)
                    section_hierarchy = get_section_hierarchy(
                        section_id, sections_cache
                    )

                    # Add section hierarchy to path in order (from root parent to current section)
                    # The hierarchy is already in correct order: [Parent Section 1, Parent Section 2, ..., Current Section]
                    for section_name_in_hierarchy in section_hierarchy:
                        if (
                            section_name_in_hierarchy
                            and section_name_in_hierarchy not in category_path
                        ):
                            category_path.append(section_name_in_hierarchy)

                    # Special handling: If section is "Most Frequently Asked Questions", use it as first category
                    # to match the original faq.json format
                    if (
                        section_name == "Most Frequently Asked Questions"
                        and category_path[0] != section_name
                    ):
                        # Reorder to put "Most Frequently Asked Questions" first
                        category_path = [section_name] + [
                            c for c in category_path if c != section_name
                        ]

                    # Get articles from section
                    page = 1
                    articles_in_section = 0
                    while True:
                        articles_url = f"{API_BASE_URL}/sections/{section_id}/articles.json?page={page}&per_page=100"
                        articles_response = requests.get(
                            articles_url, headers=HEADERS, timeout=10
                        )

                        if articles_response.status_code != 200:
                            if articles_response.status_code == 404:
                                print(f"      Section {section_id} not found (404)")
                            break

                        articles_data = articles_response.json()
                        articles = articles_data.get("articles", [])

                        if not articles:
                            break

                        print(f"      Found {len(articles)} articles on page {page}")
                        articles_in_section += len(articles)

                        for article in articles:
                            try:
                                # Extract HTML content and convert to text
                                body_html = article.get("body", "")
                                soup = BeautifulSoup(body_html, "html.parser")
                                answer = extract_text(soup)

                                article_data = {
                                    "question": article.get("title", ""),
                                    "answer": answer,
                                    "categories": category_path.copy(),
                                    "url": article.get("html_url", ""),
                                }

                                # Always add article, even if answer is empty (log it for debugging)
                                if not article_data["answer"]:
                                    print(
                                        f"        Warning: Article '{article_data['question'][:50]}...' has empty answer"
                                    )

                                all_articles.append(article_data)
                            except Exception as e:
                                print(
                                    f"        Error processing article {article.get('id', 'unknown')}: {e}"
                                )
                                continue

                        # Check for next page
                        if not articles_data.get("next_page"):
                            break

                        page += 1
                        time.sleep(0.2)  # Rate limiting

                    if articles_in_section == 0:
                        print(f"      No articles found in section {section_name}")

                except Exception as e:
                    print(
                        f"    Error processing section {section.get('id', 'unknown')}: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    continue

        print(f"\nAPI scraping complete! Found {len(all_articles)} articles")
        return all_articles

    except Exception as e:
        print(f"Error using API: {e}")
        print("Falling back to web scraping...")
        return None


def scrape_all_articles():
    """Main function to scrape all articles"""
    print("Starting scrape...")

    # Try API first (faster and more reliable)
    articles = scrape_via_api()

    if articles is not None:
        return articles

    # Fall back to web scraping
    print("\nUsing web scraping method...")
    all_articles = []

    # Get categories
    print("Fetching categories...")
    categories = get_categories()
    print(f"Found {len(categories)} categories")

    # If no categories found, try to get sections directly
    if not categories:
        print("No categories found, trying to get sections directly...")
        soup = get_soup(BASE_URL)
        if soup:
            section_links = soup.find_all(
                "a", href=re.compile(r"/hc/en-my/sections/\d+")
            )
            for link in section_links:
                href = link.get("href")
                if href:
                    full_url = urljoin(BASE_URL, href)
                    name = extract_text(link)
                    categories.append({"name": name, "url": full_url})

    # Process each category
    for idx, category in enumerate(categories, 1):
        print(f"\nProcessing category {idx}/{len(categories)}: {category['name']}")

        # Get sections in this category
        sections = get_sections(category["url"], category["name"])
        print(f"  Found {len(sections)} sections")

        if not sections:
            # Category might directly contain articles
            sections = [
                {
                    "name": category["name"],
                    "url": category["url"],
                    "category": category["name"],
                }
            ]

        # Process each section
        for section in sections:
            print(f"    Processing section: {section['name']}")
            category_path = [category["name"]]
            if section["name"] != category["name"]:
                category_path.append(section["name"])

            # Get articles from section
            article_list = get_articles_from_section(
                section["url"], section["name"], category_path
            )
            print(f"      Found {len(article_list)} articles")

            # Extract content from each article
            for article_info in article_list:
                print(f"        Extracting: {article_info['title'][:50]}...")
                article_data = extract_article_content(
                    article_info["url"],
                    article_info["title"],
                    article_info["category_path"],
                )

                if article_data and article_data["answer"]:
                    all_articles.append(article_data)

                time.sleep(0.3)  # Rate limiting

    return all_articles


if __name__ == "__main__":
    import sys
    import os

    # Ensure project root (one level up from this file) is on sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

    articles = scrape_all_articles()

    # Save to JSON file under top-level data/ for clearer separation
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, "tngd_faqs.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"\n\nScraping complete!")
    print(f"Total articles scraped: {len(articles)}")
    print(f"Saved to: {output_file}")
