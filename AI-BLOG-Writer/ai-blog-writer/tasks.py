"""Task definitions for the AI Blog Writer workflow.

This module defines the three sequential tasks:
1. Research Task: Generates a structured research brief.
2. Writing Task: Consumes the research brief via CrewAI task context and drafts the article.
3. Review Task: Consumes the draft via CrewAI task context and polishes it for publication.
"""

from crewai import Agent, Task


def create_research_task(agent: Agent) -> Task:
    """Create Task 1: Blog Topic Research and Brief Preparation.

    Args:
        agent: The Blog Research Specialist agent.

    Returns:
        Configured CrewAI Task for researching.
    """
    return Task(
        description=(
            "Perform in-depth structured research on the topic: '{topic}'.\n\n"
            "Context Parameters:\n"
            "- Target Audience: {audience}\n"
            "- Writing Tone: {tone}\n"
            "- Desired Article Length: {length}\n\n"
            "Your Responsibilities:\n"
            "1. Define the core subject matter and explain why it matters to {audience}.\n"
            "2. Identify 4 to 6 crucial subtopics, key principles, or technical concepts.\n"
            "3. Frame 3 key questions that the blog post must answer for the reader.\n"
            "4. Propose concrete examples, analogies, or practical scenarios.\n"
            "5. Flag any factual assertions or statistical claims that would require verified sources.\n"
            "   (CRITICAL: Do NOT invent fake URLs, fake studies, or fake citations).\n"
            "6. Provide a suggested structural outline for the writer."
        ),
        expected_output=(
            "A comprehensive Research Brief formatted in Markdown containing:\n"
            "- Executive Summary & Audience Alignment\n"
            "- Core Questions Answered\n"
            "- Key Concepts & Technical Breakdown\n"
            "- Practical Examples & Case Scenarios\n"
            "- Structural Blog Outline\n"
            "- Verification Notes (claims requiring factual verification in a production setting)"
        ),
        agent=agent,
    )


def create_writing_task(agent: Agent, research_task: Task) -> Task:
    """Create Task 2: First Draft Creation.

    Uses the research brief from Task 1 via CrewAI's task context mechanism.

    Args:
        agent: The Professional Blog Content Writer agent.
        research_task: The preceding research task whose output provides context.

    Returns:
        Configured CrewAI Task for drafting the blog.
    """
    return Task(
        description=(
            "Write a complete, compelling, and well-structured first draft of the blog post on '{topic}'.\n\n"
            "Workflow Context:\n"
            "- You MUST use the research brief provided by the Blog Research Specialist.\n"
            "- Target Audience: {audience}\n"
            "- Target Tone: {tone}\n"
            "- Target Length: {length}\n\n"
            "Drafting Guidelines:\n"
            "1. Craft a captivating, click-worthy H1 title.\n"
            "2. Write a strong introduction that hooks the {audience} and clearly defines the value proposition.\n"
            "3. Develop the body using clear Markdown headings (##, ###) following the research outline.\n"
            "4. Integrate real-world examples, explanations, and analogies suited to {audience}.\n"
            "5. Maintain a natural, engaging {tone} tone from start to finish.\n"
            "6. Conclude with an inspiring summary, actionable takeaways, or a thoughtful call-to-action.\n"
            "7. Aim for approximately the requested length: {length}."
        ),
        expected_output=(
            "A complete first draft of the blog post in clean Markdown format with an H1 title, "
            "introductory hook, structured section headings (H2/H3), illustrative examples, "
            "and a strong conclusion."
        ),
        agent=agent,
        context=[research_task],
    )


def create_review_task(agent: Agent, writing_task: Task) -> Task:
    """Create Task 3: Quality Review, Fact-Check, and Polish.

    Uses the draft from Task 2 via CrewAI's task context mechanism.

    Args:
        agent: The Blog Quality Reviewer and Editor agent.
        writing_task: The preceding writing task whose output provides context.

    Returns:
        Configured CrewAI Task for reviewing and polishing the final blog.
    """
    return Task(
        description=(
            "Perform a comprehensive editorial review and final polish of the drafted blog post.\n\n"
            "Context Parameters:\n"
            "- Topic: '{topic}'\n"
            "- Target Audience: {audience}\n"
            "- Expected Tone: {tone}\n"
            "- Desired Length: {length}\n\n"
            "Editorial Checklist:\n"
            "1. Mechanics: Fix any spelling, punctuation, grammar, and syntax errors.\n"
            "2. Flow & Readability: Ensure smooth transitions between paragraphs and sections.\n"
            "3. Audience Fit: Verify that technical depth matches {audience} and tone matches {tone}.\n"
            "4. Fact & Integrity Check: Eliminate unsupported exaggerations, ensure no fabricated citations "
            "or imaginary links exist.\n"
            "5. Structure & Headings: Polish headings to be punchy, descriptive, and properly hierarchical.\n"
            "6. Value Delivery: Ensure the introduction is magnetic and the conclusion provides satisfying closure.\n\n"
            "Output the final, perfected, publication-ready article in complete Markdown."
        ),
        expected_output=(
            "The final, publication-ready blog post formatted in beautiful Markdown. "
            "Includes polished H1 title, engaging introduction, clear hierarchical headings (H2, H3), "
            "refined prose, and an impactful conclusion ready for immediate publishing."
        ),
        agent=agent,
        context=[writing_task],
    )
