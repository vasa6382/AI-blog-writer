"""Unit and integration tests for the AI Blog Writer workflow.

These tests verify:
- Input validation (empty topic, unsupported dropdown options).
- Agent roles, goals, and backstories.
- Task definitions and context-dependency chaining.
- Crew assembly with sequential process.
- End-to-end execution with mocked Crew kickoff (no API credits used).
- Error handling when API key is missing or execution errors occur.
"""

from unittest.mock import MagicMock, patch
import pytest
from crewai import LLM

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
from crew import (
    validate_inputs,
    build_blog_crew,
    run_blog_generation,
    ValidationError,
    ConfigurationError,
    SUPPORTED_AUDIENCES,
    SUPPORTED_TONES,
    SUPPORTED_LENGTHS,
)


class TestInputValidation:
    """Tests for validating user inputs."""

    def test_valid_inputs_pass(self):
        """Valid inputs should pass without raising exceptions."""
        validate_inputs(
            topic="Agentic AI in Enterprise",
            audience="Software Developers",
            tone="Professional",
            length="Medium: approximately 1,000 words",
        )

    def test_empty_topic_raises_error(self):
        """Empty or whitespace-only topic should raise ValidationError."""
        with pytest.raises(ValidationError, match="Blog topic cannot be empty"):
            validate_inputs(
                topic="",
                audience="Beginners",
                tone="Friendly",
                length="Short: approximately 500 words",
            )

        with pytest.raises(ValidationError, match="Blog topic cannot be empty"):
            validate_inputs(
                topic="   ",
                audience="Students",
                tone="Educational",
                length="Short: approximately 500 words",
            )

    def test_unsupported_audience_raises_error(self):
        """Unsupported audience selection should raise ValidationError."""
        with pytest.raises(ValidationError, match="Unsupported audience"):
            validate_inputs(
                topic="Quantum Computing",
                audience="Alien Lifeforms",
                tone="Educational",
                length="Medium: approximately 1,000 words",
            )

    def test_unsupported_tone_raises_error(self):
        """Unsupported tone selection should raise ValidationError."""
        with pytest.raises(ValidationError, match="Unsupported tone"):
            validate_inputs(
                topic="Quantum Computing",
                audience="Students",
                tone="Sarcastic",
                length="Medium: approximately 1,000 words",
            )

    def test_unsupported_length_raises_error(self):
        """Unsupported length selection should raise ValidationError."""
        with pytest.raises(ValidationError, match="Unsupported length"):
            validate_inputs(
                topic="Quantum Computing",
                audience="Students",
                tone="Educational",
                length="Ultra Long: 50,000 words",
            )

    def test_all_supported_options_are_accepted(self):
        """Every option in the supported lists should be valid."""
        for audience in SUPPORTED_AUDIENCES:
            for tone in SUPPORTED_TONES:
                for length in SUPPORTED_LENGTHS:
                    validate_inputs("AI Testing", audience, tone, length)


class TestAgentCreation:
    """Tests for agent configuration and roles."""

    def test_agent_roles_and_goals(self):
        """Verify the three agents have distinct, expected roles and goals."""
        test_llm = LLM(model="gpt-4o-mini", api_key="mock-key")
        researcher = create_blog_researcher(llm=test_llm)
        writer = create_blog_writer(llm=test_llm)
        reviewer = create_blog_reviewer(llm=test_llm)

        assert researcher.role == "Blog Research Specialist"
        assert "research" in researcher.goal.lower()
        # CrewAI 1.x leaves memory as None/False when disabled
        assert not researcher.memory

        assert writer.role == "Professional Blog Content Writer"
        assert "blog post" in writer.goal.lower()

        assert reviewer.role == "Blog Quality Reviewer and Editor"
        assert "review" in reviewer.goal.lower()


class TestTaskContextAndWorkflow:
    """Tests for task creation and context passing between tasks."""

    def test_task_context_chaining(self):
        """Verify Task 2 receives Task 1 context, and Task 3 receives Task 2 context."""
        test_llm = LLM(model="gpt-4o-mini", api_key="mock-key")
        researcher = create_blog_researcher(llm=test_llm)
        writer = create_blog_writer(llm=test_llm)
        reviewer = create_blog_reviewer(llm=test_llm)

        task_research = create_research_task(agent=researcher)
        task_write = create_writing_task(agent=writer, research_task=task_research)
        task_review = create_review_task(agent=reviewer, writing_task=task_write)

        # Research task has no predecessor context
        assert not task_research.context or str(task_research.context) == "NOT_SPECIFIED"

        # Writing task must depend on Research task
        assert task_write.context == [task_research]

        # Review task must depend on Writing task
        assert task_review.context == [task_write]

    def test_build_blog_crew_structure(self):
        """Verify build_blog_crew sets up 3 agents, 3 tasks, and sequential process."""
        test_llm = LLM(model="gpt-4o-mini", api_key="mock-key")
        crew, tasks_dict = build_blog_crew(llm=test_llm)

        assert len(crew.agents) == 3
        assert len(crew.tasks) == 3
        assert str(crew.process).lower().endswith("sequential")

        assert "research_task" in tasks_dict
        assert "writing_task" in tasks_dict
        assert "review_task" in tasks_dict


class TestWorkflowExecution:
    """Tests for end-to-end workflow execution with mock LLM."""

    def test_missing_api_key_raises_configuration_error(self):
        """Running without API key or mock LLM should raise ConfigurationError."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=True):
            with pytest.raises(ConfigurationError, match="OpenAI API key not found"):
                run_blog_generation(
                    topic="Python Decorators",
                    audience="Software Developers",
                    tone="Educational",
                    length="Short: approximately 500 words",
                    api_key="",
                )

    def test_successful_mocked_crew_execution(self):
        """Verify crew.kickoff executes and extracts final blog and research brief."""
        mock_research_output = MagicMock()
        mock_research_output.raw = "# Research Brief: Microservices vs Monoliths"

        mock_crew_output = MagicMock()
        mock_crew_output.raw = "# Microservices vs Monoliths\n\nFinal polished blog post content."
        mock_crew_output.tasks_output = [mock_research_output]

        with patch("crew.Crew.kickoff", return_value=mock_crew_output) as mock_kickoff:
            result = run_blog_generation(
                topic="Microservices vs Monoliths",
                audience="Software Developers",
                tone="Professional",
                length="Medium: approximately 1,000 words",
                api_key="mock-key-for-testing",
            )

            # Check that kickoff was invoked with proper parameters
            mock_kickoff.assert_called_once()
            call_kwargs = mock_kickoff.call_args.kwargs
            assert "inputs" in call_kwargs
            assert call_kwargs["inputs"]["topic"] == "Microservices vs Monoliths"
            assert call_kwargs["inputs"]["audience"] == "Software Developers"

            # Check output dictionary structure
            assert "final_blog" in result
            assert "research_brief" in result
            assert "Final polished blog post content" in result["final_blog"]
            assert "Research Brief" in result["research_brief"]

    def test_crewai_runtime_exception_handling(self):
        """Verify that unexpected CrewAI failures are wrapped in a helpful error message."""
        with patch("crew.Crew.kickoff", side_effect=Exception("OpenAI API rate limit exceeded")):
            with pytest.raises(RuntimeError, match="CrewAI workflow execution failed"):
                run_blog_generation(
                    topic="AI Ethics",
                    audience="General Readers",
                    tone="Friendly",
                    length="Short: approximately 500 words",
                    api_key="mock-key",
                )

    def test_demo_mode_execution(self):
        """Verify demo mode generates structured outputs without requiring an API key."""
        result = run_blog_generation(
            topic="Autonomous AI Systems",
            audience="Software Developers",
            tone="Professional",
            length="Medium: approximately 1,000 words",
            demo_mode=True,
        )

        assert "final_blog" in result
        assert "research_brief" in result
        assert "Autonomous AI Systems" in result["final_blog"]
        assert "Research Brief" in result["research_brief"]
        assert "Software Developers" in result["final_blog"]

    def test_demo_mode_validates_inputs(self):
        """Verify demo mode still enforces input validation rules."""
        with pytest.raises(ValidationError, match="Blog topic cannot be empty"):
            run_blog_generation(
                topic="",
                audience="Beginners",
                tone="Friendly",
                length="Short: approximately 500 words",
                demo_mode=True,
            )


class TestUIHelpers:
    """Tests for UI helper functions in crew.py."""

    def test_calculate_metrics(self):
        """Verify word count and estimated reading time calculations."""
        from crew import calculate_metrics

        sample_text = "This is a sample blog post text containing ten words total."
        metrics = calculate_metrics(sample_text)
        assert metrics["word_count"] == 11
        assert metrics["reading_time"] >= 1

    def test_sanitize_filename(self):
        """Verify safe filename conversion."""
        from crew import sanitize_filename

        topic = "Agentic AI: The Future of Coding (2026)!"
        filename = sanitize_filename(topic)
        assert filename.endswith(".md")
        assert "!" not in filename
        assert ":" not in filename
        assert "agentic_ai" in filename



