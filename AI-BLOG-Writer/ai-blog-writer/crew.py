"""Crew orchestrator for the AI Blog Writer workflow.

This module sets up the CrewAI multi-agent sequential pipeline:
1. Validates user input parameters.
2. Initializes agents and tasks with context dependencies.
3. Assembles and executes the sequential Crew.
4. Extracts and returns the final blog and research brief.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
import os
from crewai import Crew, Process, LLM
from dotenv import load_dotenv

from agents import (
    create_blog_researcher,
    create_blog_writer,
    create_blog_reviewer,
    get_default_llm,
)
from tasks import (
    create_research_task,
    create_writing_task,
    create_review_task,
)

# Load environment configuration
load_dotenv()


def calculate_metrics(text: str) -> Dict[str, int]:
    """Calculate word count and estimated reading time from text.

    Args:
        text: The blog content string.

    Returns:
        Dict with 'word_count' and 'reading_time' (in minutes).
    """
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    reading_time_mins = max(1, round(word_count / 200))
    return {"word_count": word_count, "reading_time": reading_time_mins}


def sanitize_filename(topic: str) -> str:
    """Create a safe, sanitized Markdown filename from a blog topic.

    Args:
        topic: The blog topic title.

    Returns:
        Safe filename ending with .md.
    """
    cleaned = re.sub(r"[^\w\s-]", "", topic).strip().lower()
    cleaned = re.sub(r"[-\s]+", "_", cleaned)
    return f"{cleaned[:40] or 'blog_post'}.md"

# Supported configuration values
SUPPORTED_AUDIENCES: List[str] = [
    "Beginners",
    "Students",
    "Software Developers",
    "Business Professionals",
    "General Readers",
]

SUPPORTED_TONES: List[str] = [
    "Professional",
    "Friendly",
    "Educational",
    "Conversational",
]

SUPPORTED_LENGTHS: List[str] = [
    "Short: approximately 500 words",
    "Medium: approximately 1,000 words",
    "Long: approximately 1,500 words",
]


class ValidationError(ValueError):
    """Raised when user input validation fails."""
    pass


class ConfigurationError(RuntimeError):
    """Raised when environment or API configuration is missing or invalid."""
    pass


def validate_inputs(topic: str, audience: str, tone: str, length: str) -> None:
    """Validate all user inputs before running the CrewAI workflow.

    Args:
        topic: The blog subject string.
        audience: Target audience selection.
        tone: Tone selection.
        length: Target length selection.

    Raises:
        ValidationError: If any argument is empty or unsupported.
    """
    if not topic or not topic.strip():
        raise ValidationError("Blog topic cannot be empty. Please enter a topic.")

    if audience not in SUPPORTED_AUDIENCES:
        raise ValidationError(
            f"Unsupported audience '{audience}'. Must be one of: {', '.join(SUPPORTED_AUDIENCES)}"
        )

    if tone not in SUPPORTED_TONES:
        raise ValidationError(
            f"Unsupported tone '{tone}'. Must be one of: {', '.join(SUPPORTED_TONES)}"
        )

    if length not in SUPPORTED_LENGTHS:
        raise ValidationError(
            f"Unsupported length '{length}'. Must be one of: {', '.join(SUPPORTED_LENGTHS)}"
        )


def build_blog_crew(llm: Optional[LLM] = None) -> Tuple[Crew, Dict[str, Any]]:
    """Construct the three agents, three tasks, and the CrewAI sequential workflow.

    Args:
        llm: Configured LLM instance or None (uses default).

    Returns:
        A tuple of (Crew instance, dictionary of created tasks).
    """
    # 1. Instantiate the three agents
    researcher = create_blog_researcher(llm=llm)
    writer = create_blog_writer(llm=llm)
    reviewer = create_blog_reviewer(llm=llm)

    # 2. Instantiate tasks with sequential context passing
    task_research = create_research_task(agent=researcher)
    task_write = create_writing_task(agent=writer, research_task=task_research)
    task_review = create_review_task(agent=reviewer, writing_task=task_write)

    # 3. Assemble sequential Crew
    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[task_research, task_write, task_review],
        process=Process.sequential,
        verbose=True,
    )

    tasks_dict = {
        "research_task": task_research,
        "writing_task": task_write,
        "review_task": task_review,
    }

    return crew, tasks_dict


def generate_demo_output(topic: str, audience: str, tone: str, length: str) -> Dict[str, str]:
    """Generate realistic, structured demonstration content without LLM API calls.

    Useful for offline demos, presentations, interviews, and testing when API credits are exhausted.

    Args:
        topic: User's blog topic.
        audience: Target audience selection.
        tone: Writing style tone.
        length: Desired length option.

    Returns:
        Dict with 'final_blog' and 'research_brief'.
    """
    clean_topic = topic.strip()
    
    research_brief = (
        f"# 🔬 Research Brief: {clean_topic}\n\n"
        f"**Prepared by:** Agent 1 - Blog Research Specialist  \n"
        f"**Target Audience:** {audience}  \n"
        f"**Writing Tone:** {tone}  \n"
        f"**Length Guideline:** {length}  \n\n"
        "---\n\n"
        "### 1. Executive Summary & Value Proposition\n"
        f"'{clean_topic}' represents a vital area of interest for {audience}. "
        f"The goal of this article is to translate complex technical principles into an "
        f"engaging, actionable narrative maintaining a {tone.lower()} tone throughout.\n\n"
        "### 2. Core Questions Answered\n"
        f"1. What foundational mechanisms and paradigms underpin {clean_topic}?\n"
        f"2. How does {clean_topic} solve concrete challenges for {audience}?\n"
        "3. What practical best practices and architectural guardrails should be adopted?\n\n"
        "### 3. Key Concepts & Structural Breakdown\n"
        "- **Foundational Architecture:** Core definitions, historical progression, and primitives.\n"
        "- **Real-World Impact:** Concrete applications, industry workflows, and efficiency gains.\n"
        "- **Implementation Strategy:** Step-by-step methodologies and design patterns.\n"
        "- **Future Trajectory:** Emerging trends and strategic considerations.\n\n"
        "### 4. Illustrative Examples & Case Scenarios\n"
        f"- Practical production scenario demonstrating how {audience} achieve measurable velocity improvements.\n"
        "- Comparison framework highlighting tradeoffs between traditional workflows and modern agentic approaches.\n\n"
        "### 5. Fact-Checking & Verification Notes\n"
        "- *Audit Note:* Specific performance metrics and benchmark figures should be verified against production telemetry.\n"
    )

    final_blog = (
        f"# Mastering {clean_topic}: A Strategic Guide for {audience}\n\n"
        f"*Curated by AI Blog Writer • Target Audience: {audience} • Style: {tone}*\n\n"
        "---\n\n"
        "## Introduction: Why This Topic Matters Now\n\n"
        f"In today's rapidly evolving technological ecosystem, **{clean_topic}** has graduated from "
        f"an emerging concept into an essential competency. Whether you are building production systems, "
        f"leading engineering teams, or expanding your knowledge as part of our **{audience}** community, "
        f"mastering these principles is crucial for building resilient, future-ready solutions.\n\n"
        f"This article explores the core architecture, practical benefits, and real-world patterns of "
        f"{clean_topic}, presented with an accessible, **{tone.lower()}** voice.\n\n"
        "---\n\n"
        "## 1. Deconstructing the Core Fundamentals\n\n"
        f"At its core, **{clean_topic}** represents a profound shift in how we approach modularity, "
        "orchestration, and efficiency. Rather than relying on monolithic, brittle processes, modern "
        "implementations emphasize:\n\n"
        "- **Autonomous Task Specialization:** Breaking complex workflows into discrete, dedicated roles.\n"
        "- **Seamless Context Propagation:** Ensuring that verified research and intermediate outputs flow "
        "smoothly into downstream drafting and editing stages.\n"
        "- **Strict Quality Gates:** Incorporating automated editorial reviews to catch hallucinations, formatting "
        "flaws, and logical inconsistencies early.\n\n"
        "> *\"True scalability is achieved not by making a single component do everything, but by orchestrating "
        "specialized agents that collaborate with clear context boundaries.\"*\n\n"
        "---\n\n"
        f"## 2. Practical Applications for {audience}\n\n"
        f"How does {clean_topic} translate into real-world impact for {audience}? Consider these key advantages:\n\n"
        "### Velocity Without Sacrificing Rigor\n"
        "By dividing labor across specialized phases—research, writing, and review—teams can accelerate content and "
        "code generation cycles by over 60% while maintaining publication-grade quality.\n\n"
        "### Reduced Cognitive Load\n"
        "Instead of struggling with a blank page or unformatted output, practitioners receive structured, "
        "hierarchical Markdown articles complete with introductions, subheadings, examples, and conclusions.\n\n"
        "---\n\n"
        "## 3. Essential Best Practices & Architecture Checklist\n\n"
        f"When applying {clean_topic} in your own workflow, follow these proven best practices:\n\n"
        "1. **Define Unambiguous Personas:** Ensure each contributor has a razor-sharp goal, role, and backstory.\n"
        "2. **Chain Task Context Deterministically:** Avoid ad-hoc prompt stuffing; feed the outputs of predecessor "
        "tasks directly into successor inputs.\n"
        "3. **Incorporate Continuous Fact-Checking:** Disallow ungrounded citations and actively flag claims that require "
        "independent verification.\n"
        "4. **Iterate with Automated Tests:** Use unit test suites with mocks to ensure prompt schemas and validation "
        "rules never regress.\n\n"
        "---\n\n"
        "## Conclusion: Looking to the Future\n\n"
        f"As **{clean_topic}** continues to mature, it will redefine how technical organizations research, "
        "communicate, and deliver software. By combining disciplined agentic architecture with thoughtful human oversight, "
        f"{audience} can unlock unprecedented productivity and insight.\n\n"
        "**Key Takeaway:** High-performing content and AI systems aren't born from single monolithic prompts—they are "
        "the product of disciplined collaboration, clear delegation, and continuous editorial refinement."
    )

    return {
        "final_blog": final_blog,
        "research_brief": research_brief,
    }


def run_blog_generation(
    topic: str,
    audience: str,
    tone: str,
    length: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    custom_llm: Optional[LLM] = None,
    demo_mode: bool = False,
) -> Dict[str, str]:
    """Execute the end-to-end multi-agent blog creation pipeline.

    Args:
        topic: User's chosen blog topic.
        audience: Target audience (e.g. 'Software Developers').
        tone: Writing style tone (e.g. 'Conversational').
        length: Desired length option.
        api_key: Optional OpenAI API Key (or retrieved from env).
        model_name: Optional OpenAI model name.
        custom_llm: Optional pre-configured LLM for testing or custom provider.
        demo_mode: If True, executes offline simulation mode without consuming OpenAI API credits.

    Returns:
        Dict with keys:
            - 'final_blog': The polished Markdown blog post.
            - 'research_brief': The research brief prepared by Agent 1.

    Raises:
        ValidationError: If inputs are invalid.
        ConfigurationError: If OpenAI API key is missing and demo_mode is False.
        RuntimeError: If execution fails.
    """
    # Step 1: Validate inputs
    validate_inputs(topic=topic, audience=audience, tone=tone, length=length)

    # Step 2: Handle Demo Mode (Offline / Free Mode)
    if demo_mode:
        return generate_demo_output(topic=topic, audience=audience, tone=tone, length=length)

    # Step 3: Ensure API key is configured
    effective_api_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
    if not effective_api_key and custom_llm is None:
        raise ConfigurationError(
            "OpenAI API key not found. Please provide an API key in the UI, "
            "set OPENAI_API_KEY in your .env file, or enable Demo / Simulation Mode."
        )

    # Step 4: Configure LLM
    llm = custom_llm or get_default_llm(api_key=effective_api_key, model_name=model_name)

    # Step 5: Build Crew and Tasks
    crew, tasks_dict = build_blog_crew(llm=llm)

    # Step 6: Execute sequential pipeline with inputs
    inputs = {
        "topic": topic.strip(),
        "audience": audience,
        "tone": tone,
        "length": length,
    }

    try:
        crew_output = crew.kickoff(inputs=inputs)
    except Exception as exc:
        raise RuntimeError(f"CrewAI workflow execution failed: {str(exc)}") from exc

    # Step 7: Extract outputs
    final_blog = str(crew_output.raw) if hasattr(crew_output, "raw") else str(crew_output)

    # Extract research brief
    research_brief = ""
    if hasattr(crew_output, "tasks_output") and crew_output.tasks_output:
        research_brief = str(crew_output.tasks_output[0].raw)
    elif getattr(tasks_dict["research_task"], "output", None):
        research_brief = str(tasks_dict["research_task"].output.raw)
    else:
        research_brief = "Research brief details integrated directly into final draft."

    return {
        "final_blog": final_blog,
        "research_brief": research_brief,
    }

