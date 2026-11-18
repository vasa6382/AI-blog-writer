"""Agent definitions for the AI Blog Writer application.

This module encapsulates the creation of the three CrewAI agents:
1. Blog Researcher: Synthesizes core concepts, outlines, and factual requirements.
2. Blog Writer: Crafts an engaging, structured first draft adhering to tone and audience.
3. Blog Reviewer: Edits, refines, verifies consistency, and produces publication-ready content.
"""

from typing import List, Optional
import os
from crewai import Agent, LLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_default_llm(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> LLM:
    """Instantiate and configure the LLM for CrewAI agents.

    Args:
        api_key: Optional OpenAI API key override. Defaults to OPENAI_API_KEY env var.
        model_name: Optional model name. Defaults to OPENAI_MODEL_NAME or 'gpt-4o-mini'.

    Returns:
        Configured crewai.LLM instance.
    """
    resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
    resolved_model = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    
    # In CrewAI, LLM can be configured with model and api_key
    return LLM(
        model=resolved_model,
        api_key=resolved_key,
        temperature=0.7,
    )


def create_blog_researcher(
    llm: Optional[LLM] = None,
    tools: Optional[List] = None,
) -> Agent:
    """Create Agent 1: The Blog Research Specialist.

    Responsibilities:
    - Analyze the topic and target audience.
    - Extract key concepts, subtopics, and fundamental questions.
    - Assemble a structured research brief.
    - Note claims that need source verification without inventing citations.

    Args:
        llm: LLM instance for this agent.
        tools: Optional list of external tools (e.g., search tools for future expansion).

    Returns:
        Configured CrewAI Agent for research.
    """
    return Agent(
        role="Blog Research Specialist",
        goal=(
            "Thoroughly analyze the topic '{topic}' for the target audience '{audience}'. "
            "Identify key concepts, logical subtopics, and essential questions, and synthesize "
            "them into an actionable, structured research brief with an outline. Flag any "
            "claims requiring external verification without fabricating citations."
        ),
        backstory=(
            "You are an analytical research specialist with a talent for breaking down complex "
            "topics into clear, digestible conceptual frameworks. You pride yourself on intellectual "
            "honesty and accuracy. You never fabricate web citations, make up statistics, or claim "
            "to have accessed live internet data without a verified research tool. Instead, you rely "
            "on rigorous logical reasoning, foundational domain knowledge, and clear structural thinking."
        ),
        tools=tools or [],
        llm=llm,
        verbose=True,
        memory=False,
    )


def create_blog_writer(
    llm: Optional[LLM] = None,
    tools: Optional[List] = None,
) -> Agent:
    """Create Agent 2: The Professional Blog Content Writer.

    Responsibilities:
    - Ingest the research brief from the Researcher.
    - Create a compelling title and hook.
    - Structure clear headings and subheadings.
    - Include real-world examples, explanations, and conclusion.
    - Maintain target tone and audience alignment.

    Args:
        llm: LLM instance for this agent.
        tools: Optional list of external tools.

    Returns:
        Configured CrewAI Agent for writing.
    """
    return Agent(
        role="Professional Blog Content Writer",
        goal=(
            "Craft an engaging, high-quality, and comprehensive blog post on '{topic}' based on "
            "the research brief provided. Tailor the content specifically to the '{audience}' "
            "audience, adopt a '{tone}' tone, and adhere to the target length '{length}'. "
            "Use clear Markdown formatting, compelling headings, and illustrative examples."
        ),
        backstory=(
            "You are an award-winning digital content creator and professional blog writer. "
            "You have a knack for turning research briefs into vivid, flowing narratives that "
            "captivate readers from the opening line. You structure your articles logically "
            "with clean Markdown formatting, intuitive subheadings, concrete real-world examples, "
            "and memorable takeaways. You strictly respect the requested tone, audience background, "
            "and word length."
        ),
        tools=tools or [],
        llm=llm,
        verbose=True,
        memory=False,
    )


def create_blog_reviewer(
    llm: Optional[LLM] = None,
    tools: Optional[List] = None,
) -> Agent:
    """Create Agent 3: The Blog Quality Reviewer and Editor.

    Responsibilities:
    - Check grammar, spelling, readability, and coherence.
    - Spot repetition, awkward phrasing, and unsupported claims.
    - Verify adherence to target audience, tone, and formatting.
    - Polish introduction, transitions, and conclusion.
    - Return the final, publication-ready Markdown blog post.

    Args:
        llm: LLM instance for this agent.
        tools: Optional list of external tools.

    Returns:
        Configured CrewAI Agent for review.
    """
    return Agent(
        role="Blog Quality Reviewer and Editor",
        goal=(
            "Perform a rigorous editorial review of the drafted blog post. Check for grammatical "
            "precision, clarity, logical progression, tone consistency ({tone}), and audience "
            "suitability ({audience}). Polish weak sections, refine headings, eliminate fluff, "
            "and output the final, publication-ready Markdown article without introducing "
            "fabricated claims or fake citations."
        ),
        backstory=(
            "You are a seasoned senior editor and publication director known for meticulous standards. "
            "You elevate good writing into exceptional, publication-ready journalism. You verify that "
            "the draft directly addresses the core topic, flows naturally, and maintains consistent "
            "stylistic tone. You eliminate redundant filler while preserving the author's distinct voice, "
            "ensuring every heading and transition enhances reader engagement."
        ),
        tools=tools or [],
        llm=llm,
        verbose=True,
        memory=False,
    )
