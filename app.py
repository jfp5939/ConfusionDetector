import os
import json
import time
import feedparser
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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

# ── Mock patient reports (shown in the UI as examples) ───────────────────────
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
    """Generate plain-English patient-friendly summaries for a batch of papers."""
    if not papers:
        return []

    abstracts_block = "\n\n".join(
        f"[{i+1}] {p['abstract']}" for i, p in enumerate(papers)
    )
    prompt = (
        "You are a compassionate doctor explaining medical research to patients with no medical background.\n"
        "For each numbered abstract below, write a plain-English summary in 40 words or fewer that:\n"
        "- Explains what the research found\n"
        "- Uses zero medical jargon\n"
        "- Feels reassuring and human\n\n"
        "Return ONLY valid JSON:\n"
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
    except Exception as e:
        print("Summary error:", e)
        return [" ".join(p["abstract"].split()[:40]) + "…" for p in papers]


def generate_paper_analysis(abstract):
    """Generate treatment plan + questions for a single paper."""
    prompt = (
        "You are a kind, knowledgeable doctor explaining a medical research paper to a patient.\n\n"
        "Based on this abstract, provide:\n"
        "1. A treatment plan in plain English — what this research suggests patients can do or expect. "
        "Use simple everyday language. Be specific and actionable.\n"
        "2. Three questions the patient should ask their doctor at their next appointment.\n\n"
        "Return ONLY valid JSON:\n"
        '{"treatment_plan": "2-3 sentences in plain English", '
        '"questions_for_doctor": ["question 1", "question 2", "question 3"]}\n\n'
        f"Abstract:\n{abstract}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return {
            "treatment_plan": data.get("treatment_plan", ""),
            "questions_for_doctor": data.get("questions_for_doctor", [])[:3],
        }
    except Exception as e:
        print("Analysis error:", e)
        return {"treatment_plan": "", "questions_for_doctor": []}


def analyze_patient_report(report_text):
    """Analyze a user-submitted medical report in plain English."""
    prompt = (
        "You are a compassionate doctor helping a patient understand their medical report.\n\n"
        "Read this medical report and explain it as if talking to the patient directly:\n"
        "1. What does this report say in plain English?\n"
        "2. What should they focus on or be aware of?\n"
        "3. Three specific questions they should ask their doctor.\n\n"
        "Be warm, clear, and avoid all medical jargon. Do NOT diagnose — only explain and empower.\n\n"
        "Return ONLY valid JSON:\n"
        '{"plain_explanation": "2-3 sentences", '
        '"focus_points": ["point 1", "point 2", "point 3"], '
        '"questions_for_doctor": ["question 1", "question 2", "question 3"]}\n\n'
        f"Medical Report:\n{report_text}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return {
            "plain_explanation": data.get("plain_explanation", ""),
            "focus_points": data.get("focus_points", [])[:3],
            "questions_for_doctor": data.get("questions_for_doctor", [])[:3],
        }
    except Exception as e:
        print("Report analysis error:", e)
        return {"plain_explanation": "", "focus_points": [], "questions_for_doctor": []}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           categories=CATEGORIES,
                           default_categories=DEFAULT_CATEGORIES,
                           mock_reports=MOCK_REPORTS)


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
            " ".join(paper["abstract"].split()[:40]) + "…"
        paper["summary"] = summary
        paper["category_label"] = CATEGORIES.get(category, category)

    CACHE[cache_key] = {"papers": papers, "fetched_at": now}
    return jsonify({"papers": papers, "category": category, "start": start})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Treatment plan + doctor questions for a single paper, fetched on expand."""
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
    ANALYSIS_CACHE[paper_id] = {"data": result, "fetched_at": now}
    return jsonify(result)


@app.route("/api/report", methods=["POST"])
def api_report():
    """Analyze a patient-submitted medical report."""
    body = request.get_json(silent=True) or {}
    report_text = body.get("report", "").strip()

    if not report_text or len(report_text) < 20:
        return jsonify({"error": "Please enter a valid medical report."}), 400

    result = analyze_patient_report(report_text)
    return jsonify(result)


@app.route("/api/mock_reports")
def api_mock_reports():
    return jsonify({"reports": MOCK_REPORTS})


if __name__ == "__main__":
    app.run(debug=True, port=8080)