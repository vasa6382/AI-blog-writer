# ✍️ AI Blog Writer (Multi-Agent System with CrewAI)

An autonomous, multi-agent AI application built with **CrewAI**, **OpenAI**, and **Streamlit** that takes a blog topic and collaboratively conducts research, writes a structured first draft, and performs an editorial quality review before outputting a publication-ready Markdown article.

---

## 📌 Project Overview

Writing high-quality blog content requires distinct skills:
1. **Research & Structuring:** Uncovering core concepts, key questions, and an outline.
2. **Writing & Storytelling:** Crafting a catchy title, compelling hook, engaging body, and clear examples.
3. **Editing & Fact-Checking:** Polishing grammar, ensuring tone consistency, eliminating fluff, and verifying that claims are grounded.

Rather than forcing a single LLM prompt to do everything at once, **AI Blog Writer** uses an **Agentic AI architecture** with three specialized agents collaborating in a sequential pipeline.

```
┌─────────────────────────────────┐
│     User Input via Streamlit    │
│  (Topic, Audience, Tone, Length)│
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Agent 1: Blog Researcher      │
│  Produces: Research Brief (.md) │
└────────────────┬────────────────┘
                 │ Task Context
                 ▼
┌─────────────────────────────────┐
│   Agent 2: Blog Writer          │
│  Produces: Full 1st Draft (.md) │
└────────────────┬────────────────┘
                 │ Task Context
                 ▼
┌─────────────────────────────────┐
│   Agent 3: Blog Reviewer        │
│  Produces: Polished Blog (.md)  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    Interactive Preview & Export │
│   (Markdown render & .md file)  │
└─────────────────────────────────┘
```

---

## 📁 Project Structure

```
ai-blog-writer/
├── app.py                  # Modern Streamlit UI for user interaction
├── crew.py                 # Multi-agent workflow assembly & execution logic
├── agents.py               # CrewAI agent definitions (roles, goals, backstories)
├── tasks.py                # CrewAI task definitions with context-passing
├── requirements.txt        # Production Python dependencies
├── .env.example            # Environment template for API keys
├── .gitignore              # Git ignore rules for security and virtual environments
├── README.md               # Setup, architectural documentation, and usage guide
└── tests/
    └── test_blog_workflow.py # Comprehensive test suite (input validation, workflow, mocks)
```

---

## 🛠️ Technology Stack

- **Python (>= 3.10)**: Core language
- **CrewAI (>= 1.15)**: Multi-agent orchestration framework
- **OpenAI API**: Large Language Model backend (`gpt-4o-mini`, `gpt-4o`)
- **Streamlit**: Web user interface with reactive status indicators and live preview
- **python-dotenv**: Environment configuration management
- **Pytest**: Automated testing framework with mock support

---

## ⚙️ Quickstart Setup Guide (Windows)

Follow these step-by-step instructions to set up and run the project locally on Windows:

### 1. Open Terminal and Navigate to Project Directory

Open **PowerShell** or **Command Prompt**:

```powershell
cd "c:\Users\Vasanth\OneDrive - MISSIONMIND IT SERVICES PRIVATE LIMITED\Documents\CREWAI\ai-blog-writer"
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

```powershell
.venv\Scripts\activate
```

> **Note:** If PowerShell displays an execution policy error (`script execution is disabled on this system`), run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> and re-run `.venv\Scripts\activate`.

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

*(Note for Python 3.14 users: If your environment uses Python 3.14+, pip might require `--ignore-requires-python` if third-party packages specify `<3.14`).*

### 5. Configure Your OpenAI API Key

Copy the `.env.example` file to `.env`:

```powershell
copy .env.example .env
```

Open `.env` in any text editor and paste your OpenAI API key:

```env
OPENAI_API_KEY=sk-proj-yourActualOpenAIApiKeyHere
OPENAI_MODEL_NAME=gpt-4o-mini
```

> You can also enter or override your API key directly in the Streamlit sidebar!

### 6. Launch the Streamlit Web App

```powershell
streamlit run app.py
```

Streamlit will automatically open your default browser at `http://localhost:8501`.

---

## 💡 Core CrewAI Concepts Explained

For a beginner or interviewer, here is what each CrewAI concept means:

| Concept | Explanation | Real-World Analogy |
| :--- | :--- | :--- |
| **Agent** | An autonomous unit programmed with a specific persona, `role`, `goal`, and `backstory`. | An employee with a job description. |
| **Task** | A concrete assignment given to an Agent, specifying a `description` and `expected_output`. | A ticket or assignment given to an employee. |
| **Context** | Mechanism (`context=[prior_task]`) to pass the output of a prior task into the next task. | Handing off a brief to the next teammate. |
| **Tool** | A function or capability an Agent can call (e.g., search Google, read files). | A calculator or web browser. |
| **Process** | The orchestration rule (`Process.sequential` or `Process.hierarchical`). | Standard operating procedure / workflow. |
| **Crew** | The container bringing agents, tasks, and process together to execute (`kickoff()`). | The cross-functional project team. |

---

## 🔄 How the 3 Agents Collaborate

1. **Blog Research Specialist (Agent 1):**
   - Receives the user's `topic`, `audience`, `tone`, and `length`.
   - Analyzes core concepts, key questions, and outlines the article structure.
   - Outputs a Markdown **Research Brief**.
   - *Key Rule:* Never hallucinates search citations without a real tool.

2. **Professional Blog Content Writer (Agent 2):**
   - Takes the **Research Brief** via CrewAI task context (`context=[task_research]`).
   - Drafts the full blog post with an H1 title, hook, headings (H2/H3), examples, and conclusion.
   - Adheres to the user's selected tone and audience background.

3. **Blog Quality Reviewer and Editor (Agent 3):**
   - Takes the first draft via CrewAI task context (`context=[task_write]`).
   - Audits grammar, structural flow, readability, and fact integrity.
   - Refines headings, eliminates repetitive filler, and outputs the final publication-ready Markdown.

---

## 🧪 Running Automated Tests

All tests are designed to execute quickly without consuming OpenAI API credits using mocks:

```powershell
python -m pytest tests/test_blog_workflow.py -v
```

### What the test suite covers:
- ✅ **Input Validation**: Empty topic rejection and supported dropdown options.
- ✅ **Agent Initialization**: Correct roles, goals, and backstories.
- ✅ **Task Context Chaining**: Verifies `writing_task.context == [research_task]` and `review_task.context == [writing_task]`.
- ✅ **Crew Assembly**: Sequential process verification.
- ✅ **Mocked End-to-End Run**: Verifies `crew.kickoff()` parameter delivery and output extraction.
- ✅ **Error Handling**: Graceful failure messages when API key is missing or API errors occur.
- ✅ **UI Helpers**: Word count, reading time, and filename sanitization.

---

## 🌐 Extending with Live Web Search (Future Roadmap)

Currently, the Researcher uses LLM reasoning and internal knowledge base to ensure reproducibility without requiring paid search APIs.

To add live internet search:
1. Install CrewAI tools: `pip install crewai-tools`
2. Obtain a Serper API key from [serper.dev](https://serper.dev/) (free tier available).
3. In `agents.py`, import the search tool:
   ```python
   from crewai_tools import SerperDevTool
   search_tool = SerperDevTool()
   ```
4. Attach `tools=[search_tool]` to `create_blog_researcher(tools=[search_tool])`.
5. The Researcher will automatically search Google, retrieve live URLs, and incorporate verified real-world links into the brief!

---

## 🚀 Deployment Options

1. **Streamlit Community Cloud (Easiest)**:
   - Push repository to GitHub.
   - Connect to [share.streamlit.io](https://share.streamlit.io).
   - Add `OPENAI_API_KEY` to Streamlit Secrets.
2. **Docker Container**:
   - Create a `Dockerfile` with `FROM python:3.11-slim`, `COPY . .`, `RUN pip install -r requirements.txt`, and `CMD ["streamlit", "run", "app.py"]`.
3. **Cloud VMs (AWS EC2 / Azure VM / GCP Cloud Run)**:
   - Deploy as a containerized service with an Nginx reverse proxy.
