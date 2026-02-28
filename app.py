import os
import json
import time
import urllib.request
import urllib.error
import feedparser
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("\n" + "="*60)
    print("  WARNING: GEMINI_API_KEY is not set.")
    print("  AI features will fall back to plain text snippets.")
    print("  Get a FREE key at: aistudio.google.com")
    print("  Then run: export GEMINI_API_KEY='...'")
    print("="*60 + "\n")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY
)

CATEGORIES = {
    "q-bio.QM":   "General Medicine",
    "q-bio.GN":   "Genetics",
    "q-bio.NC":   "Neuroscience",
    "q-bio.TO":   "Tissue & Organs",
    "q-bio.CB":   "Cell Biology",
}

DEFAULT_CATEGORIES = ["q-bio.QM", "q-bio.GN", "q-bio.NC"]

CACHE = {}
ANALYSIS_CACHE = {}
CACHE_TTL = 30 * 60  # 30 minutes

MOCK_REPORTS = [
    {
        "id": "mock_1",
        "label": "Type 2 Diabetes Report",
        "text": "Patient: 54yo female. HbA1c: 8.2%. Fasting glucose: 178 mg/dL. BMI: 31. Currently on Metformin 1000mg twice daily. Mild peripheral neuropathy noted. Kidney function normal. BP: 138/88."
    },
    {
        "id": "mock_2",
        "label": "Hypertension & Heart Report",
        "text": "Patient: 62yo male. BP: 158/96 mmHg. LDL: 142 mg/dL. HDL: 38 mg/dL. EF: 52%. On Lisinopril 10mg, Atorvastatin 40mg. No chest pain. Mild shortness of breath on exertion."
    },
    {
        "id": "mock_3",
        "label": "Anxiety & Depression Report",
        "text": "Patient: 29yo female. PHQ-9 score: 14 (moderate depression). GAD-7 score: 12 (moderate anxiety). Sleep: 4-5 hrs/night. Currently not on medication. No suicidal ideation. Seeking therapy options."
    },
]


def call_gemini(prompt):
    """Call Gemini 1.5 Flash (free tier) and return the text response."""
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4}
    }).encode("utf-8")
    req = urllib.request.Request(
        GEMINI_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["candidates"][0]["content"]["parts"][0]["text"].strip()


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


def generate_patient_summaries(papers):
    """Generate plain-English summaries for a batch of papers using Gemini."""
    if not papers:
        return []
    if not GEMINI_API_KEY:
        return [" ".join(p["abstract"].split()[:50]) + "…" for p in papers]

    abstracts_block = "\n\n".join(
        f"[{i+1}] {p['abstract']}" for i, p in enumerate(papers)
    )
    prompt = (
        "You are explaining medical research to someone with no medical background.\n"
        "For each numbered abstract below, write ONE sentence (max 40 words) in very simple, "
        "everyday English that explains what the research found. No jargon at all.\n\n"
        "Return ONLY valid JSON, no extra text:\n"
        '{"summaries": ["sentence 1", "sentence 2", ...]}\n\n'
        f"Abstracts:\n{abstracts_block}"
    )
    try:
        text = call_gemini(prompt)
        # Strip markdown code fences if present
        text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(text)
        summaries = data.get("summaries", [])
        while len(summaries) < len(papers):
            summaries.append(None)
        return summaries
    except Exception as e:
        print("Summary error:", e)
        return [" ".join(p["abstract"].split()[:50]) + "…" for p in papers]


def generate_paper_analysis(abstract):
    """Generate plain-English takeaways + doctor questions for a paper."""
    if not GEMINI_API_KEY:
        return {
            "treatment_plan": "Get a free Gemini API key at aistudio.google.com, then set GEMINI_API_KEY to enable plain-English analysis.",
            "questions_for_doctor": [
                "Does this research apply to my condition?",
                "Are there new treatments based on findings like these?",
                "Should I make any lifestyle changes based on this?",
            ],
        }

    prompt = (
        "You are a kind doctor explaining a research paper to a patient in plain English.\n\n"
        "Based on this abstract:\n"
        "1. In 2-3 simple sentences, explain what this research means for everyday people — "
        "what they can do, expect, or know. No medical jargon.\n"
        "2. List 3 questions the patient should ask their doctor.\n\n"
        "Return ONLY valid JSON, no extra text:\n"
        '{"treatment_plan": "2-3 plain sentences", '
        '"questions_for_doctor": ["q1", "q2", "q3"]}\n\n'
        f"Abstract:\n{abstract}"
    )
    try:
        text = call_gemini(prompt)
        text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(text)
        return {
            "treatment_plan": data.get("treatment_plan", ""),
            "questions_for_doctor": data.get("questions_for_doctor", [])[:3],
        }
    except Exception as e:
        print("Analysis error:", e)
        return {"error": str(e), "treatment_plan": "", "questions_for_doctor": []}


def analyze_patient_report(report_text, papers=None):
    """Explain a medical report in plain English using Gemini."""
    if not GEMINI_API_KEY:
        return {
            "error": "Get a free Gemini API key at aistudio.google.com and set GEMINI_API_KEY.",
            "plain_explanation": "", "focus_points": [], "questions_for_doctor": []
        }

    prompt = (
        "You are a compassionate doctor helping a patient understand their medical report.\n\n"
        "Read this report and explain it directly to the patient:\n"
        "1. What does this say in plain English? (2-3 warm, simple sentences — no jargon)\n"
        "2. Three things the patient should focus on or watch out for.\n"
        "3. Three questions to ask their doctor.\n\n"
        "Do NOT diagnose. Be warm and clear.\n\n"
        "Return ONLY valid JSON, no extra text:\n"
        '{"plain_explanation": "2-3 sentences", '
        '"focus_points": ["p1", "p2", "p3"], '
        '"questions_for_doctor": ["q1", "q2", "q3"]}\n\n'
        f"Medical Report:\n{report_text}"
    )
    try:
        text = call_gemini(prompt)
        text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(text)
        return {
            "plain_explanation": data.get("plain_explanation", ""),
            "focus_points": data.get("focus_points", [])[:3],
            "questions_for_doctor": data.get("questions_for_doctor", [])[:3],
        }
    except Exception as e:
        print("Report analysis error:", e)
        return {"error": str(e), "plain_explanation": "", "focus_points": [], "questions_for_doctor": []}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           categories=CATEGORIES,
                           default_categories=DEFAULT_CATEGORIES,
                           mock_reports=MOCK_REPORTS)


@app.route("/api/health")
def api_health():
    return jsonify({"openai_configured": bool(GEMINI_API_KEY)})


@app.route("/api/feed")
def api_feed():
    category = request.args.get("category", "q-bio.QM")
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
    summaries = generate_patient_summaries(papers)

    for i, paper in enumerate(papers):
        summary = summaries[i] if i < len(summaries) and summaries[i] else \
            " ".join(paper["abstract"].split()[:50]) + "…"
        paper["summary"] = summary
        paper["category_label"] = CATEGORIES.get(category, category)

    CACHE[cache_key] = {"papers": papers, "fetched_at": now}
    return jsonify({"papers": papers, "category": category, "start": start})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    body = request.get_json(silent=True) or {}
    paper_id = body.get("id", "").strip()
    abstract = body.get("abstract", "").strip()

    if not abstract:
        return jsonify({"treatment_plan": "", "questions_for_doctor": []})

    now = time.time()
    if paper_id in ANALYSIS_CACHE:
        entry = ANALYSIS_CACHE[paper_id]
        if now - entry["fetched_at"] < CACHE_TTL:
            return jsonify(entry["data"])

    result = generate_paper_analysis(abstract)
    if "error" not in result:
        ANALYSIS_CACHE[paper_id] = {"data": result, "fetched_at": now}
    return jsonify(result)


@app.route("/api/report", methods=["POST"])
def api_report():
    body = request.get_json(silent=True) or {}
    report_text = body.get("report", "").strip()
    papers = body.get("papers", [])

    if not report_text or len(report_text) < 20:
        return jsonify({"error": "Please enter a valid medical report (at least 20 characters)."}), 400

    result = analyze_patient_report(report_text, papers)
    return jsonify(result)


@app.route("/api/mock_reports")
def api_mock_reports():
    return jsonify({"reports": MOCK_REPORTS})


@app.route("/sw.js")
def service_worker():
    return send_from_directory("static", "sw.js", mimetype="application/javascript")


if __name__ == "__main__":
    app.run(debug=True, port=8080)
