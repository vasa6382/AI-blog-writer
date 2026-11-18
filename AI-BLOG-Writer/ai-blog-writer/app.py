"""AI Blog Writer - Streamlit User Interface.

A modern, intuitive UI powered by CrewAI to autonomously research, draft,
and polish high-caliber blog posts using a 3-agent collaborative pipeline.
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

from crew import (
    run_blog_generation,
    calculate_metrics,
    sanitize_filename,
    ValidationError,
    ConfigurationError,
    SUPPORTED_AUDIENCES,
    SUPPORTED_TONES,
    SUPPORTED_LENGTHS,
)

# Load existing environment variables
load_dotenv()


def main():
    """Main execution function for the Streamlit web application."""
    # Page configuration
    st.set_page_config(
        page_title="AI Blog Writer | Powered by CrewAI",
        page_icon="✍️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for modern styling
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.3rem;
            font-weight: 800;
            background: linear-gradient(120deg, #2563EB, #7C3AED, #DB2777);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.05rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .agent-card {
            background-color: #F8FAFC;
            border-radius: 8px;
            padding: 0.9rem;
            border: 1px solid #E2E8F0;
            margin-bottom: 0.6rem;
        }
        .agent-title {
            font-weight: 700;
            color: #1E293B;
            font-size: 0.95rem;
        }
        .agent-desc {
            color: #64748B;
            font-size: 0.82rem;
            margin-top: 0.2rem;
        }
        .metric-badge {
            display: inline-block;
            background: #EEF2FF;
            color: #4338CA;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }
        .stButton>button {
            background: linear-gradient(90deg, #2563EB, #4F46E5);
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 0.65rem 1.4rem;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: Configuration & Agent Architecture
    with st.sidebar:
        st.image(
            "https://raw.githubusercontent.com/joaomdmoura/crewAI/main/docs/crewai_logo.png",
            width=150,
        )
        st.markdown("### ⚙️ Settings & API Key")

        env_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=env_api_key,
            type="password",
            help="Reads from .env by default. Enter your OpenAI API key (sk-...) here if not set in .env.",
            placeholder="sk-proj-...",
        )

        model_option = st.selectbox(
            "OpenAI Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0,
            help="gpt-4o-mini offers fast execution and high reasoning quality.",
        )

        st.markdown("---")
        mode_selection = st.radio(
            "Workflow Execution Mode",
            options=["🧪 Demo / Simulation Mode (Free - No API credits)", "⚡ Live Multi-Agent Crew (OpenAI API)"],
            index=0,
            help="Demo Mode generates realistic multi-agent outputs without using API credits. Live Mode executes autonomous CrewAI agents using your OpenAI API key.",
        )
        demo_mode = "Demo" in mode_selection

        if demo_mode:
            st.info("💡 **Demo Mode Active:** Tests full multi-agent workflow, preview, and download without API credits.")
        elif api_key_input:
            st.success("✅ OpenAI API Key configured")
        else:
            st.warning("⚠️ OpenAI API Key required for Live Mode")

        st.markdown("---")
        st.markdown("### 🤖 Collaborative Crew")
        st.markdown(
            """
            <div class="agent-card">
                <div class="agent-title">1. Blog Researcher</div>
                <div class="agent-desc">Explores core concepts, frames critical questions, and builds a structured research brief.</div>
            </div>
            <div class="agent-card">
                <div class="agent-title">2. Blog Writer</div>
                <div class="agent-desc">Consumes research context to draft an engaging article with titles, headings, and examples.</div>
            </div>
            <div class="agent-card">
                <div class="agent-title">3. Blog Reviewer</div>
                <div class="agent-desc">Refines grammar, readability, verifies facts, and ensures tone and audience alignment.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Crew Process: **Sequential Workflow** with task context propagation.")

    # Main Application Area
    st.markdown('<div class="main-title">AI Blog Writer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Multi-Agent Content Creation Pipeline powered by CrewAI and OpenAI</div>',
        unsafe_allow_html=True,
    )

    # Input Form
    with st.container():
        topic_input = st.text_input(
            "Blog Topic",
            placeholder="e.g. How Autonomous AI Agents Will Revolutionize Software Engineering",
            help="Enter the subject or theme you want the AI agents to research and write about.",
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            audience_selection = st.selectbox(
                "Target Audience",
                options=SUPPORTED_AUDIENCES,
                index=2,  # Default: Software Developers
                help="Tailors language complexity, context depth, and examples.",
            )

        with col2:
            tone_selection = st.selectbox(
                "Writing Tone",
                options=SUPPORTED_TONES,
                index=0,  # Default: Professional
                help="Dictates the voice and style of the generated post.",
            )

        with col3:
            length_selection = st.selectbox(
                "Desired Length",
                options=SUPPORTED_LENGTHS,
                index=1,  # Default: Medium (~1,000 words)
                help="Approximate length guideline provided to the writing and editing agents.",
            )

        generate_button = st.button("🚀 Generate Blog Post", use_container_width=True)

    # Execution and Result Handling
    if generate_button:
        if not topic_input.strip():
            st.error("❌ Please provide a blog topic before generating.")
        elif not demo_mode and not api_key_input.strip():
            st.error(
                "❌ OpenAI API Key is missing! Please provide your API key in the sidebar, "
                "set `OPENAI_API_KEY` in your `.env` file, or toggle Demo Mode above."
            )
        else:
            mode_label = "Demo Simulation" if demo_mode else "CrewAI Multi-Agent Pipeline"
            status_box = st.status(f"🤖 Running {mode_label}...", expanded=True)
            try:
                status_box.write("🔍 **Phase 1:** Blog Researcher is formulating the research brief and outline...")
                if demo_mode:
                    time.sleep(0.7)
                status_box.write("✍️ **Phase 2:** Blog Writer is crafting the first draft using research context...")
                if demo_mode:
                    time.sleep(0.7)
                status_box.write("🧐 **Phase 3:** Blog Quality Reviewer is auditing, polishing, and finalizing...")
                if demo_mode:
                    time.sleep(0.6)

                with st.spinner("Collaborative crew is processing... Please wait a moment."):
                    results = run_blog_generation(
                        topic=topic_input,
                        audience=audience_selection,
                        tone=tone_selection,
                        length=length_selection,
                        api_key=api_key_input.strip(),
                        model_name=model_option,
                        demo_mode=demo_mode,
                    )

                status_box.update(
                    label="✅ Blog Generation Complete! All agents finished successfully.",
                    state="complete",
                    expanded=False,
                )

                # Store in session state for persistence
                st.session_state["final_blog"] = results["final_blog"]
                st.session_state["research_brief"] = results["research_brief"]
                st.session_state["generated_topic"] = topic_input

            except ValidationError as val_err:
                status_box.update(label="❌ Validation Error", state="error", expanded=False)
                st.error(f"Input validation error: {str(val_err)}")
            except ConfigurationError as cfg_err:
                status_box.update(label="❌ Configuration Error", state="error", expanded=False)
                st.error(f"Configuration error: {str(cfg_err)}")
            except Exception as exc:
                status_box.update(label="❌ Generation Failed", state="error", expanded=False)
                st.error(f"An unexpected error occurred during execution: {str(exc)}")

    # Display Generated Output
    if "final_blog" in st.session_state and st.session_state["final_blog"]:
        final_content = st.session_state["final_blog"]
        brief_content = st.session_state.get("research_brief", "")
        current_topic = st.session_state.get("generated_topic", "blog_post")

        metrics = calculate_metrics(final_content)

        st.markdown("---")
        st.markdown("### 📄 Generated Publication-Ready Blog")

        # Metrics banner
        col_m1, col_m2, col_m3 = st.columns([2, 2, 4])
        with col_m1:
            st.markdown(f'<span class="metric-badge">📊 {metrics["word_count"]} Words</span>', unsafe_allow_html=True)
        with col_m2:
            st.markdown(f'<span class="metric-badge">⏱️ ~{metrics["reading_time"]} Min Read</span>', unsafe_allow_html=True)
        with col_m3:
            st.download_button(
                label="💾 Download Blog Post (.md)",
                data=final_content,
                file_name=sanitize_filename(current_topic),
                mime="text/markdown",
                use_container_width=True,
            )

        # Render Final Blog Markdown
        st.markdown("#### Final Article Preview")
        st.markdown(final_content)

        # Expandable Research Brief
        with st.expander("🔍 View Research Brief (Generated by Agent 1: Blog Researcher)"):
            if brief_content:
                st.markdown(brief_content)
            else:
                st.info("The research brief was synthesized directly into the writing draft.")

        # Additional download option for Research Brief
        if brief_content:
            st.download_button(
                label="📥 Download Research Brief (.md)",
                data=brief_content,
                file_name=f"research_brief_{sanitize_filename(current_topic)}",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
