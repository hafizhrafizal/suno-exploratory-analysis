"""
Suno Discord Chat Analysis — TXT File Input with Chunking & Batching
=====================================================================
Usage:
    pip install openai reportlab
    export OPENAI_API_KEY="sk-..."
    python suno_analysis.py suno_chat_2023_08.txt
    python suno_analysis.py suno_chat_2023_08.txt --model gpt-5-mini
    python suno_analysis.py suno_chat_2024_04.txt --token-limit 120000
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Union

from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from xml.sax.saxutils import escape


# ── Config ────────────────────────────────────────────────────
DEFAULT_TOKEN_LIMIT = 500_000
DEFAULT_MODEL = "gpt-5.4"
CHARS_PER_TOKEN = 4.0

DEFAULT_SYSTEM_PROMPT = """
[Role]
You are an expert analyst of online community dynamics.
You will analyze a provided chunk of Discord chat logs from the Suno AI music community.
Use only the provided logs. Do not use external knowledge. Do not infer facts that are not directly supported by the text.

[Core Constraints]
- Only analyze content explicitly present in the logs
- Do not speculate about user identity, intent, or context beyond the text
- Every claim must be supported by at least one direct excerpt
- If evidence is insufficient, state: "Not enough evidence in the logs"
"""

DEFAULT_USER_PROMPT = """
[Instructions]
1. Identify Topics of Contention
- Extract all relevant distinct topics where conflict or tension is present
- A topic must include at least one of the following:
  * disagreement between users
  * complaints or criticism
  * repeated friction across messages
- Do not include neutral discussions
- Focus on topics related to ethical, moral, legal issues related to the use of AI music generation
- Identify all time range that discuss the topic, and summarize how the discussion evolved over time

2. Identify Key Participants (non-Suno Team) per Topic
- Select all users (exhaustive) who:
  * contribute multiple times OR
  * influence the direction of the discussion OR
  * give opinion to the related topic
- For each participant, extract:
  * Idea: what they are saying
  * Stance: their position (e.g., supportive, critical, neutral)
  * Attitude: tone (e.g., frustrated, defensive, sarcastic, calm)
- Use only observable textual cues (e.g., wording, repetition, emphasis)
3. Identify the Suno Team's Position/Response per Topic
- Extract any relevant messages from Suno team members related to the topic
- If no response is present, state: "No response from Suno team in the logs"
4. Extract Evidence and Explain
- Provide verbatim excerpts from the logs
- Keep quotes short but representative
- Link each excerpt clearly to the topic and participant
- Follow each excerpt with a concise explanation of how it supports the analysis

[Output Format]
Produce structured markdown exactly in the following format:

## Topic: <Concise Topic Name>

### Description
<Clear, detailed and evidence-based explanation of the contention>

### Timeline
<list of all time ranges that discuss the topic>
<summary of how the discussion evolved over time>

### Key Participants
**<Username>**
  - Idea: <comprehensive idea explanation grounded in text>
  - Stance: <supportive / critical / neutral>
  - Attitude: <tone based on textual cues>
  - Evidence:
    - "<excerpt>"
    - Explanation: <how this supports the interpretation>
### Suno Team Response
**<Username>**
- <Summary of Suno team messages related to the topic, if any>
- Evidence:
    - "<excerpt>"
### Additional Evidence (Optional)
"<excerpt>"
*Explanation*:
"""


# ── Token counting ────────────────────────────────────────────
def count_tokens_approx(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


# ── TXT file parsing & chunking ──────────────────────────────
LINE_PATTERN = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\s")


def parse_date_from_line(line: str) -> str | None:
    m = LINE_PATTERN.match(line)
    return m.group(1) if m else None


def load_txt_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def chunk_chat_text(text: str, token_limit: int) -> list[dict]:
    """Split text into chunks at line breaks, respecting token limit."""
    total_tokens = count_tokens_approx(text)
    char_limit = int(token_limit * CHARS_PER_TOKEN)

    if total_tokens <= token_limit:
        lines = text.strip().split("\n")
        start_date = end_date = None
        for line in lines:
            d = parse_date_from_line(line)
            if d:
                if start_date is None:
                    start_date = d
                end_date = d
        return [{
            "text": text,
            "start_date": start_date or "unknown",
            "end_date": end_date or "unknown",
            "token_count": total_tokens,
        }]

    lines = text.split("\n")
    chunks = []
    current_lines = []
    current_size = 0
    chunk_start_date = chunk_end_date = None

    for line in lines:
        line_len = len(line) + 1
        if current_size + line_len > char_limit and current_lines:
            chunk_text = "\n".join(current_lines)
            chunks.append({
                "text": chunk_text,
                "start_date": chunk_start_date or "unknown",
                "end_date": chunk_end_date or "unknown",
                "token_count": count_tokens_approx(chunk_text),
            })
            current_lines = []
            current_size = 0
            chunk_start_date = chunk_end_date = None

        current_lines.append(line)
        current_size += line_len
        d = parse_date_from_line(line)
        if d:
            if chunk_start_date is None:
                chunk_start_date = d
            chunk_end_date = d

    if current_lines:
        chunk_text = "\n".join(current_lines)
        chunks.append({
            "text": chunk_text,
            "start_date": chunk_start_date or "unknown",
            "end_date": chunk_end_date or "unknown",
            "token_count": count_tokens_approx(chunk_text),
        })

    return chunks


# ── OpenAI API ────────────────────────────────────────────────
def call_openai_model(system_prompt, user_prompt, chat_data, model=DEFAULT_MODEL):
    client = OpenAI()
    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt}\n\nChat Data:\n{chat_data}"},
        ],
    )

    elapsed = time.time() - start
    usage = response.usage

    return {
        "text": response.choices[0].message.content.strip(),
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "elapsed_seconds": elapsed,
    }


# ── PDF export ────────────────────────────────────────────────
def md_bold_to_rl(text: str) -> str:
    parts = text.split("**")
    if len(parts) == 1:
        return escape(text)
    out = []
    for i, part in enumerate(parts):
        safe = escape(part)
        out.append(f"<b>{safe}</b>" if i % 2 == 1 else safe)
    if len(parts) % 2 == 0:
        out.append("**")
    return "".join(out)


def line_to_paragraph(line: str, styles) -> Union[Paragraph, Spacer]:
    stripped = line.strip()
    if not stripped:
        return Spacer(1, 0.15 * inch)
    if stripped.startswith("### "):
        return Paragraph(escape(stripped[4:]), styles["Heading3"])
    if stripped.startswith("## "):
        return Paragraph(escape(stripped[3:]), styles["Heading2"])
    if stripped.startswith("# "):
        return Paragraph(escape(stripped[2:]), styles["Heading1"])
    formatted = md_bold_to_rl(stripped)
    try:
        return Paragraph(formatted, styles["Normal"])
    except Exception:
        return Paragraph(escape(stripped), styles["Normal"])


def export_to_pdf(content: str, pdf_path: str):
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = [line_to_paragraph(line, styles) for line in content.split("\n")]
    doc.build(story)
    print(f"  PDF saved: {pdf_path}")


# ── Report builder ────────────────────────────────────────────
def build_report(batch_results, filename, model, partial=False, error=None):
    lines = [
        f"# Suno Community Analysis: {filename}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    if partial:
        lines.append("## PARTIAL ANALYSIS (INTERRUPTED)")
        if error:
            lines.append(f"Error: {error}")
        lines.append(f"Completed {len(batch_results)} of expected chunks.")
        lines.append("")

    date_range = f"{batch_results[0]['start_date']} to {batch_results[-1]['end_date']}"

    lines.append(f"**Date range:** {date_range}")
    lines.append(f"**Chunks processed:** {len(batch_results)}")
    lines.append(f"**Model:** {model}")
    lines.append("")

    for batch in batch_results:
        if len(batch_results) > 1:
            lines.append(f"## Chunk {batch['chunk_num']}: "
                         f"{batch['start_date']} to {batch['end_date']}")
            lines.append("")
        lines.append(batch["response"])
        lines.append("")

    return "\n".join(lines)


# ── Main pipeline ─────────────────────────────────────────────
def analyze_chat_file(
    filepath, system_prompt=DEFAULT_SYSTEM_PROMPT, user_prompt=DEFAULT_USER_PROMPT,
    token_limit=DEFAULT_TOKEN_LIMIT, model=DEFAULT_MODEL, output_dir="summary_output",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    filename = Path(filepath).stem

    # Load and chunk
    print(f"\n{'='*60}")
    print(f"ANALYZING: {filepath}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    text = load_txt_file(filepath)
    total_tokens = count_tokens_approx(text)
    print(f"Total: {len(text):,} chars (~{total_tokens:,} tokens)")

    chunks = chunk_chat_text(text, token_limit)
    print(f"Chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  {i+1}: {c['start_date']} to {c['end_date']} (~{c['token_count']:,} tokens)")

    # Process chunks
    batch_results = []
    total_input = 0
    total_output = 0

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1}/{len(chunks)} ---")
        chunk_label = f"chunk {i+1}/{len(chunks)}, {chunk['start_date']} to {chunk['end_date']}"

        try:
            result = call_openai_model(
                system_prompt=system_prompt,
                user_prompt=f"{user_prompt} ({chunk_label})",
                chat_data=chunk["text"],
                model=model,
            )

            batch_results.append({
                "chunk_num": i + 1,
                "start_date": chunk["start_date"],
                "end_date": chunk["end_date"],
                "response": result["text"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "elapsed": result["elapsed_seconds"],
            })

            total_input += result["input_tokens"]
            total_output += result["output_tokens"]

            print(f"  Done in {result['elapsed_seconds']:.1f}s | "
                  f"In: {result['input_tokens']:,} | Out: {result['output_tokens']:,}")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(f"  ERROR: {error_msg}")

            if any(kw in str(e).lower() for kw in
                   ["billing", "quota", "insufficient", "exceeded", "credit"]):
                print("  BILLING/QUOTA ERROR — stopping early")

            if batch_results:
                print(f"\n  Saving {len(batch_results)} completed chunk(s)...")
                partial = build_report(batch_results, filename, model, partial=True, error=error_msg)
                export_to_pdf(partial, str(out_dir / f"{filename}_PARTIAL.pdf"))
                
            raise

    # Done — save outputs
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(batch_results)} chunk(s)")
    print(f"Tokens — In: {total_input:,} | Out: {total_output:,}")
    print(f"{'='*60}")

    report = build_report(batch_results, filename, model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    pdf_path = out_dir / f"{filename}_summary.pdf"
    export_to_pdf(report, str(pdf_path))

    
    return batch_results


# ── CLI ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Analyze Suno Discord chat logs for conflicts, key persons, and events."
    )
    parser.add_argument("txtfile", help="Path to the .txt chat export file")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--token-limit", type=int, default=DEFAULT_TOKEN_LIMIT,
                        help=f"Max tokens per chunk (default: {DEFAULT_TOKEN_LIMIT})")
    parser.add_argument("--system-prompt", default=None,
                        help="Path to a .txt file with custom system prompt")
    parser.add_argument("--output-dir", default="summary_output",
                        help="Output directory (default: summary_output)")

    args = parser.parse_args()

    if not os.path.exists(args.txtfile):
        print(f"ERROR: File not found: {args.txtfile}")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY first")
        sys.exit(1)

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt:
        if os.path.exists(args.system_prompt):
            with open(args.system_prompt) as f:
                system_prompt = f.read().strip()
            print(f"\nLoaded system prompt: {args.system_prompt}")
        else:
            print(f"WARNING: {args.system_prompt} not found, using default")

    try:
        analyze_chat_file(
            filepath=args.txtfile,
            system_prompt=system_prompt,
            token_limit=args.token_limit,
            model=args.model,
            output_dir=args.output_dir,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()