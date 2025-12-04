import os
from litellm import completion
from .registry import ToolRegistry, Tool
from .webscraping import webscraping


def _format_scrape_result(scrape_result: dict) -> str:
    if not isinstance(scrape_result, dict):
        return str(scrape_result)
    parts = []
    title = scrape_result.get("title")
    if title:
        parts.append(f"Title: {title}")
    main = scrape_result.get("main_content")
    if main:
        parts.append("\nMain Content:\n" + main.strip())
    formatted = scrape_result.get("formatted_content")
    if formatted:
        parts.append("\nFormatted Content:\n" + formatted.strip())
    paras = scrape_result.get("paragraphs") or []
    if paras:
        parts.append("\nParagraphs:\n" + "\n".join(p.strip() for p in paras if p.strip()))
    return "\n".join(parts).strip() or str(scrape_result)


def save_file(task_description=None, content: str = None, file_path: str = "log/agent_output.txt", append: bool = False,
              features=None, elements_filter=None, branching_factor=None, playwright_manager=None, log_folder='log',
              model_name: str = None, **_kwargs):
    """
    Save text to a file. If content is not provided, scrape the current page and let the model format it.
    """
    # Build context from content or scrape
    context_text = content
    if context_text is None:
        scrape = webscraping(task_description or "Scrape current page before saving",
                             features=features,
                             elements_filter=elements_filter,
                             branching_factor=branching_factor,
                             playwright_manager=playwright_manager,
                             log_folder=log_folder)
        context_text = _format_scrape_result(scrape)

    target_path = file_path or "log/agent_output.txt"
    target_path = os.path.expanduser(target_path)

    # Trim context and format via LLM
    trimmed_context = (context_text or "")[:6000]
    final_text = context_text
    summary_model = model_name or "gpt-4o-mini"
    completion_kwargs = {}
    if "gemini" in summary_model.lower():
        completion_kwargs["custom_llm_provider"] = "gemini"
        completion_kwargs["api_key"] = os.getenv("GEMINI_API_KEY")

    extension = os.path.splitext(target_path)[1].lower()
    style_hint = (
        "Produce valid CSV (with a header row) strictly following CSV rules. Avoid HTML/DOM noise. "
        "Keep cells compact, no newlines inside cells. If page is blank or inaccessible, write a short note row."
    ) if extension == ".csv" else (
        "Produce a clean, concise write-up that fits the user intent. Avoid HTML/DOM noise. "
        "If page is blank or inaccessible, say so."
    )

    try:
        resp = completion(
            model=summary_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a formatter. Given source text, rewrite it into a well-structured output appropriate "
                        "for the target file. Avoid raw HTML/DOM noise. "
                        f"{style_hint}"
                    ),
                },
                {
                    "role": "user",
                    "content": trimmed_context,
                },
            ],
            max_tokens=800,
            **completion_kwargs,
        )
        final_text = resp.choices[0].message.content.strip()
    except Exception as e:
        return {"status": "error", "file_path": target_path, "message": f"formatting_failed: {str(e)}"}

    # Write result
    directory = os.path.dirname(target_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    mode = "a" if append else "w"
    with open(target_path, mode, encoding="utf-8") as f:
        f.write(final_text)
        if append and not final_text.endswith("\n"):
            f.write("\n")
    return {"status": "success", "file_path": target_path}


def register_save_file_tool():
    ToolRegistry.register(Tool(
        name="save_file",
        func=save_file,
        description=(
            "Save provided text to a file. If content is omitted, scrape the current page, "
            "have the model reformat it (summary/CSV/etc.), and then write it."
        ),
        parameters={
            "content": {
                "type": "string",
                "description": "Optional text to write. If omitted, the tool will scrape and format the current page.",
            },
            "file_path": {
                "type": "string",
                "description": "Path to the file (relative or absolute). Defaults to log/agent_output.txt",
            },
            "append": {
                "type": "boolean",
                "description": "Append instead of overwrite (default: false)",
            }
        }
    ))
