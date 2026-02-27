import os
import json
import time
import feedparser
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CATEGORIES = {
    "cs.AI": "AI",
    "cs.LG": "Machine Learning",
    "cs.CL": "NLP",
    "cs.CV": "Computer Vision",
    "stat.ML": "ML (Stats)",
    "math.ST": "Statistics",
    "math.NA": "Numerical Analysis",
}

DEFAULT_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "math.ST"]

CACHE = {}
PREREQ_CACHE = {}
CACHE_TTL = 30 * 60  # 30 minutes


def fetch_arxiv_papers(category, start=0, max_results=15):
    url = (
        f"http://export.arxiv.org/api/query"
        f"?search_query=cat:{category}"
        f"&sortBy=submittedDate&sortOrder=descending"
        f"&start={start}&max_results={max_results}"
    )
    feed = feedparser.parse(url)
    papers = []
    for entry in feed.entries:
        arxiv_id = entry.get("id", "")
        if "abs" not in arxiv_id:
            arxiv_id = arxiv_id.replace("http://arxiv.org/", "http://arxiv.org/abs/")
        authors = [a.name for a in entry.get("authors", [])]
        papers.append({
            "id": entry.get("id", ""),
            "title": entry.get("title", "").replace("\n", " ").strip(),
            "authors": authors,
            "abstract": entry.get("summary", "").replace("\n", " ").strip(),
            "published": entry.get("published", "")[:10],
            "arxiv_url": arxiv_id,
        })
    return papers


def generate_summaries(papers):
    """Batch-generate 30-word summaries for a list of papers. Fast — summaries only."""
    if not papers:
        return []

    abstracts_block = "\n\n".join(
        f"[{i+1}] {p['abstract']}" for i, p in enumerate(papers)
    )
    prompt = (
        "You are a science communicator. For each numbered abstract below, write a "
        "plain-English summary in exactly 30 words or fewer that a curious non-expert "
        "could understand. Return ONLY valid JSON:\n"
        '{"summaries": ["summary 1", "summary 2", ...]}\n\n'
        f"Abstracts:\n{abstracts_block}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        summaries = data.get("summaries", [])
        while len(summaries) < len(papers):
            summaries.append(None)
        return summaries
    except Exception:
        return [" ".join(p["abstract"].split()[:30]) + "…" for p in papers]


def generate_prerequisites(abstract):
    """Generate prerequisites for a single paper abstract. Called on-demand."""
    prompt = (
        "You are explaining a research paper to someone with no science background.\n"
        "List exactly 3-4 things the reader needs to understand BEFORE reading this paper.\n"
        "Write each as ONE simple sentence in everyday language, as if talking to a curious "
        "12-year-old. Do NOT just name a concept — explain what it actually means.\n\n"
        "Bad: 'Linear algebra'\n"
        "Good: 'You need to know that computers can store information in grids of numbers "
        "and do math on millions of them at once.'\n\n"
        "Return ONLY valid JSON:\n"
        '{"prerequisites": ["sentence 1", "sentence 2", "sentence 3"]}\n\n'
        f"Abstract:\n{abstract}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("prerequisites", [])[:4]
    except Exception:
        return []


@app.route("/")
def index():
    return render_template("index.html", categories=CATEGORIES, default_categories=DEFAULT_CATEGORIES)


@app.route("/api/feed")
def api_feed():
    category = request.args.get("category", "cs.AI")
    try:
        start = int(request.args.get("start", 0))
    except ValueError:
        start = 0

    cache_key = (category, start)
    now = time.time()

    if cache_key in CACHE:
        entry = CACHE[cache_key]
        if now - entry["fetched_at"] < CACHE_TTL:
            return jsonify({"papers": entry["papers"], "category": category, "start": start})

    papers = fetch_arxiv_papers(category, start=start)
    summaries = generate_summaries(papers)

    for i, paper in enumerate(papers):
        summary = summaries[i] if i < len(summaries) and summaries[i] else \
            " ".join(paper["abstract"].split()[:30]) + "…"
        paper["summary"] = summary
        paper["category_label"] = CATEGORIES.get(category, category)

    CACHE[cache_key] = {"papers": papers, "fetched_at": now}
    return jsonify({"papers": papers, "category": category, "start": start})


@app.route("/api/prerequisites", methods=["POST"])
def api_prerequisites():
    """Prerequisites for a single paper, fetched on demand when user expands a card."""
    body = request.get_json(silent=True) or {}
    paper_id = body.get("id", "").strip()
    abstract = body.get("abstract", "").strip()

    if not abstract:
        return jsonify({"prerequisites": []})

    now = time.time()
    if paper_id in PREREQ_CACHE:
        entry = PREREQ_CACHE[paper_id]
        if now - entry["fetched_at"] < CACHE_TTL:
            return jsonify({"prerequisites": entry["prerequisites"]})

    prompt = (
        "Read this research paper abstract carefully.\n\n"
        "Identify exactly 3 things the author ASSUMES the reader already knows — "
        "look at the terminology they use without explaining it, the concepts they treat as obvious, "
        "and the background knowledge they take for granted.\n\n"
        "For each assumption, write ONE sentence that explains what it means in simple everyday "
        "language, as if talking to a curious 12-year-old with no science background. "
        "Start each sentence with 'You need to know that…'\n\n"
        "Return ONLY valid JSON:\n"
        '{"prerequisites": ["You need to know that...", "You need to know that...", "You need to know that..."]}\n\n'
        f"Abstract:\n{abstract}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        prereqs = data.get("prerequisites", [])[:3]
    except Exception:
        prereqs = []

    PREREQ_CACHE[paper_id] = {"prerequisites": prereqs, "fetched_at": now}
    return jsonify({"prerequisites": prereqs})


@app.route("/sw.js")
def service_worker():
    response = app.send_static_file("sw.js")
    response.headers["Content-Type"] = "application/javascript"
    response.headers["Service-Worker-Allowed"] = "/"
    return response


if __name__ == "__main__":
    app.run(debug=True, port=8080)
