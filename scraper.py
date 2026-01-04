import os
import requests
from bs4 import BeautifulSoup

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "AgenticRAGBot/1.0 (educational project)"
}


def scrape_wikipedia(topic: str) -> str:
    topic_clean = topic.strip().replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{topic_clean}"

    response = requests.get(url, headers=HEADERS, allow_redirects=True)

    if response.status_code != 200:
        raise ValueError(
            f"Wikipedia page not found for topic: '{topic}'. "
            f"Try a more specific topic name."
        )

    soup = BeautifulSoup(response.text, "html.parser")

    # Detect disambiguation pages
    if soup.find("table", {"id": "disambigbox"}):
        raise ValueError(
            f"'{topic}' is ambiguous on Wikipedia. "
            f"Please use a more specific topic name."
        )

    content_div = soup.find("div", {"id": "mw-content-text"})
    paragraphs = content_div.find_all("p")

    text = ""
    for p in paragraphs:
        if p.text.strip():
            text += p.text.strip() + "\n"

    if len(text.strip()) < 500:
        raise ValueError(
            f"Not enough meaningful content found for '{topic}'."
        )

    file_path = os.path.join(DATA_DIR, f"{topic_clean}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    return file_path
