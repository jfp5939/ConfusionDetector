import os
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are an expert tutor who helps people understand confusing research text.

Given a research paragraph, respond with EXACTLY this JSON structure:
{
  "confusing_parts": ["...", "..."],
  "assumed_background": ["...", "..."],
  "clarification_questions": ["...", "..."],
  "what_to_learn_next": ["...", "..."]
}

Be concise. Each list should have 2-4 items. Return only valid JSON, no extra text."""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )

    result = response.choices[0].message.content
    import json
    return jsonify(json.loads(result))


if __name__ == "__main__":
    app.run(debug=True)
