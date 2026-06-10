"""
memory_helper.py — Reconstructs and logs decisions post-mortem.

Uses local Ollama chat models to analyze git diffs, commits, or Antigravity
transcripts, structures them into decision logs, and updates the knowledge vault.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import click
import httpx

# Resolve REPO_KNOWLEDGE/src path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from repo_knowledge.config import OLLAMA_URL
from repo_knowledge.knowledge import KnowledgeService

# Preferred chat models in order of capability/presence
PREFERRED_MODELS = [
    "qwen2.5-coder:7b",
    "deepseek-r1:8b",
    "llama3.1:8b",
    "gemma4:e2b",
    "qwen2.5:3b",
    "llama3.2:3b",
]


def get_available_chat_model() -> str:
    """Query Ollama and return the best available chat model."""
    try:
        r = httpx.get(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=5.0)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        for pref in PREFERRED_MODELS:
            if pref in models:
                return pref
            # Check for partial matches (e.g. without tags)
            for m in models:
                if m.startswith(pref.split(":")[0]):
                    return m
        # Fallback to any model that isn't embedding if possible
        chat_models = [m for m in models if "embed" not in m]
        if chat_models:
            return chat_models[0]
        if models:
            return models[0]
    except Exception:
        pass
    return "qwen2.5-coder:7b"  # Default fallback guess


def run_git(args: list[str]) -> str:
    """Run a git command in the current directory and return stdout."""
    try:
        res = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="ignore",
        )
        return res.stdout.strip()
    except subprocess.SubprocessError as e:
        raise RuntimeError(f"Git command failed: {e}")


def query_llm(model: str, system_prompt: str, user_prompt: str) -> dict:
    """Send a query to Ollama's chat endpoint and return the parsed JSON response."""
    url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 16384, "num_predict": 2048},
    }

    try:
        click.echo(f"  [LLM] Querying {model}...")
        r = httpx.post(url, json=payload, timeout=60.0)
        r.raise_for_status()
        response_text = r.json()["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Ollama chat API call failed: {e}")

    # Remove DeepSeek thinking block if present
    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)

    # Find outer JSON block using first { and last }
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(f"LLM did not output a valid JSON block. Output was:\n{response_text}")

    json_str = response_text[start : end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON block: {e}\nBlock:\n{json_str}")


SYSTEM_PROMPT = """You are a software engineering decision archivist.
Your job is to extract technical decisions from changes,
logs, or transcripts.
Analyze the provided information and output a single JSON block representing the decision.

Your output MUST be exactly a JSON object conforming to this schema
(do not include markdown wrapping outside the JSON):
{
  "topic": "slugified-topic-name (e.g. 'embedding_model', 'auth_handling')",
  "name": "slugified_entry_name (e.g. 'upgrade_to_qwen')",
  "description": "Short sentence explaining what was done.",
  "options_considered": [
    {
      "name": "Option Name",
      "status": "SELECTED" or "REJECTED",
      "rationale": "Short explanation of pros/cons or rejection reasoning."
    }
  ],
  "rationale": "Detailed explanation of why the final choice was selected."
}

Do not include any conversational filler. Return ONLY the raw JSON object."""


@click.command()
@click.option("--diff", is_flag=True, help="Analyze unstaged and staged workspace changes.")
@click.option("--commits", default=0, help="Analyze the last N commits.")
@click.option(
    "--transcript", is_flag=True, help="Find and parse the latest Antigravity session transcript."
)
@click.option(
    "--topic", default=None, help="Override topic name (forces all entries into this topic)."
)
def main(diff: bool, commits: int, transcript: bool, topic: str | None) -> None:
    click.secho("=== REPO_KNOWLEDGE Decision Memory Helper ===", fg="cyan", bold=True)

    if not (diff or commits > 0 or transcript):
        click.echo(
            "Error: Please select at least one source: --diff, --commits <N>, or --transcript"
        )
        sys.exit(1)

    model = get_available_chat_model()
    click.echo(f"Using chat model: {model}")

    source_text = ""

    if diff:
        click.echo("Gathering local git changes...")
        # Capture staged and unstaged changes with minimal context
        staged_stat = run_git(["diff", "--cached", "--stat"])
        unstaged_stat = run_git(["diff", "--stat"])
        staged = run_git(["diff", "--cached", "-U1"])
        unstaged = run_git(["diff", "-U1"])
        source_text += f"=== GIT DIFF STAT ===\n{staged_stat}\n{unstaged_stat}\n\n=== GIT DIFF STAGED ===\n{staged}\n\n=== GIT DIFF UNSTAGED ===\n{unstaged}\n"  # noqa: E501

    if commits > 0:
        click.echo(f"Gathering the last {commits} commits...")
        commits_log = run_git(["log", "-n", str(commits), "-p", "-U1"])
        source_text += f"=== GIT COMMIT LOGS ===\n{commits_log}\n"

    # Defensively truncate to prevent blowing context window or hitting output limits
    if len(source_text) > 8000:
        click.secho(
            "  [WARN] Source text truncated to 8000 chars for LLM analysis", fg="yellow", err=True
        )
        source_text = source_text[:8000] + "\n... [SOURCE TRUNCATED TO SAVE TOKENS] ...\n"

    if transcript:
        click.echo("Searching for latest Antigravity transcript...")
        # Check standard AppData location
        appdata = os.getenv("APPDATA", "")
        if not appdata:
            click.secho("  [WARNING] APPDATA environment variable not found.", fg="yellow")
        else:
            brain_dir = Path(appdata) / "../.gemini/antigravity/brain"
            brain_dir = brain_dir.resolve()
            if not brain_dir.exists():
                click.secho(
                    f"  [WARNING] Antigravity brain directory not found at: {brain_dir}",
                    fg="yellow",
                )
            else:
                # Find the most recently modified transcript.jsonl under brain/
                transcripts = list(brain_dir.glob("**/transcript.jsonl"))
                if not transcripts:
                    click.secho(
                        "  [WARNING] No transcript.jsonl logs found under brain/.", fg="yellow"
                    )
                else:
                    transcripts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    latest_t = transcripts[0]
                    click.echo(
                        f"  Found latest log: {latest_t.name} (modified: {latest_t.stat().st_mtime})"  # noqa: E501
                    )
                    try:
                        # Extract last 100 lines to avoid blowing context
                        lines = latest_t.read_text(encoding="utf-8").splitlines()
                        recent_lines = lines[-100:]
                        parsed_log = ""
                        for line in recent_lines:
                            try:
                                data = json.loads(line)
                                if "content" in data:
                                    parsed_log += (
                                        f"[{data.get('type', 'CHAT')}]: {data['content']}\n"
                                    )
                            except Exception:
                                pass
                        source_text += f"=== RECENT CHAT LOG ===\n{parsed_log}\n"
                    except OSError as e:
                        click.secho(f"  [ERROR] Could not read transcript: {e}", fg="red")

    if not source_text.strip() or len(source_text.strip()) < 20:
        click.secho(
            "No significant source data gathered to extract decisions from. Exiting.", fg="yellow"
        )
        sys.exit(0)

    click.echo("\nExtracting decisions using local LLM...")
    try:
        decision_data = query_llm(model, SYSTEM_PROMPT, source_text)
    except Exception as e:
        click.secho(f"LLM extraction failed: {e}", fg="red")
        sys.exit(1)

    # Apply overrides
    target_topic = topic or decision_data.get("topic")
    if not target_topic:
        click.secho("Error: No topic generated by LLM and no --topic override provided.", fg="red")
        sys.exit(1)

    # Initialize Service
    svc = KnowledgeService()

    click.echo(f"Logging decision '{decision_data.get('name')}' under topic '{target_topic}'...")
    res = svc.log_decision(
        topic=target_topic,
        name=decision_data.get("name", "auto_entry"),
        description=decision_data.get("description", ""),
        rationale=decision_data.get("rationale", ""),
        options_considered=decision_data.get("options_considered"),
    )

    if "error" in res:
        click.secho(f"Error saving decision: {res['error']}", fg="red")
        sys.exit(1)
    else:
        click.secho(f"[OK] {res['message']}", fg="green")


if __name__ == "__main__":
    main()
