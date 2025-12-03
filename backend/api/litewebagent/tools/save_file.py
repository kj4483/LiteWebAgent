import os
from litellm import completion
from .registry import ToolRegistry, Tool
from .webscraping import webscraping


def _format_scrape_result(scrape_result: dict) -> str:
    """Create a readable text blob from the webscraping result."""
    if not isinstance(scrape_result, dict):
        return str(scrape_result)

    lines = []
    title = scrape_result.get("title")
    if title:
        lines.append(f"Title: {title}")

    main_content = scrape_result.get("main_content")
    if main_content:
        lines.append("\nMain Content:")
        lines.append(main_content.strip())

    formatted = scrape_result.get("formatted_content")
    if formatted:
        lines.append("\nFormatted Content:")
        lines.append(formatted.strip())

    paragraphs = scrape_result.get("paragraphs") or []
    if paragraphs:
        lines.append("\nParagraphs:")
        lines.extend([p.strip() for p in paragraphs if p.strip()])

    return "\n".join(lines) or str(scrape_result)


async def save_file(
    content: str = None,
    file_path: str = "log/agent_output.txt",
    append: bool = False,
    features=None,
    branching_factor=None,
    playwright_manager=None,
    log_folder='log',
    s3_path=None,
    elements_filter=None,
    model_name: str = None,
):
    """
    Save text to a file. If content is not provided, scrape the current page to build context,
    then let the model generate a clean, task-appropriate output (e.g., summary, notes, CSV, or other formats)
    suitable for the requested file path. Always write model-generated text, not raw scraped blobs.
    Creates parent directories if needed.
    """
    # Build context: prefer provided content; otherwise scrape page.
    context_text = content
    if context_text is None:
        scrape = await webscraping(
            task_description="Scrape current page before saving",
            features=features,
            branching_factor=branching_factor,
            playwright_manager=playwright_manager,
            log_folder=log_folder,
            s3_path=s3_path,
            elements_filter=elements_filter,
        )
        context_text = _format_scrape_result(scrape)
        
        # If scraping returned minimal content, also capture page structure via DOM
        if not context_text or "Main content not found" in context_text:
            try:
                page = await playwright_manager.get_page()
                # Get page's full HTML structure for better context
                page_html = await page.content()
                # Try to extract meaningful text from different selectors
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(page_html, 'html.parser')
                
                # Try multiple content selectors
                selectors_to_try = [
                    'main', 'article', '[role="main"]', '.content', '.main-content', 
                    '[class*="content"]', 'body'
                ]
                
                additional_text = []
                for selector in selectors_to_try:
                    if selector.startswith('.'):
                        elements = soup.select(selector)
                    elif selector.startswith('['):
                        elements = soup.select(selector)
                    else:
                        elements = soup.find_all(selector)
                    
                    for elem in elements:
                        text = elem.get_text(strip=True)
                        if text and len(text) > 50:  # Only include substantial text
                            additional_text.append(text[:2000])  # Limit per element
                            break
                    if additional_text:
                        break
                
                if additional_text:
                    context_text = "\n".join(additional_text)
            except Exception as e:
                pass  # Fall back to original scrape result

    target_path = file_path or "log/agent_output.txt"
    target_path = os.path.expanduser(target_path)
    # Generate final text with the model, tailored to the file type.
    final_text = context_text
    # Trim context to avoid huge prompts
    max_context_chars = 8000
    trimmed_context = (context_text or "")[:max_context_chars]

    try:
        summary_model = model_name or "gpt-4o-mini"
        # summary_model = "gpt-4o-mini"
        extension = os.path.splitext(target_path)[1].lower()
        
        if extension == ".csv":
            # Special handling for CSV with page structure focus
            system_message = (
                "You are a data extractor. Given page content, extract meaningful information and output it as valid CSV. "
                "Include a header row with column names like: Component, Description, Type, Details, etc. "
                "Extract key page elements, sections, content blocks, and their descriptions. "
                "Each row should describe a distinct part of the page structure or content. "
                "Follow CSV rules strictly: no newlines in cells, proper escaping. "
                "If page is blank/inaccessible, still provide at least one descriptive row."
            )
        else:
            system_message = (
                "You are a formatter. Given source text, rewrite it into a well-structured output appropriate "
                "for the target file. Avoid raw HTML/DOM noise. "
                "Produce a clean, concise write-up that fits the user intent. "
                "If page is blank or inaccessible, say so."
            )

        completion_kwargs = {}
        if "gemini" in summary_model.lower():
            completion_kwargs["custom_llm_provider"] = "gemini"
            completion_kwargs["api_key"] = os.getenv("GEMINI_API_KEY")

        resp = completion(
            model=summary_model,
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": trimmed_context if trimmed_context else "(No page content available - page may not have loaded)",
                },
            ],
            max_tokens=1200,
            **completion_kwargs,
        )
        final_text = resp.choices[0].message.content.strip()
    except Exception as e:
        # Signal failure and avoid writing bad content
        return {"status": "error", "file_path": target_path, "message": f"formatting_failed: {str(e)}"}

    # If the model returned something that looks like an error, avoid writing it.
    if final_text.lower().startswith("formatting failed"):
        return {"status": "error", "file_path": target_path, "message": final_text}

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
            "Save provided text content to a file on disk. If content is omitted, first scrape the current page, "
            "then generate a clean, task-appropriate output (summary, notes, CSV, etc.) using the model. "
            "Avoid dumping raw scraped HTML/DOM. If file_path ends with .csv, output valid CSV text (header + rows) following CSV rules."
        ),
        parameters={
            "content": {
                "type": "string",
                "description": "The text content to write. If omitted, the tool will scrape the current page and generate a clean, task-appropriate output (summary, notes, CSV, etc.) using the model.",
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
