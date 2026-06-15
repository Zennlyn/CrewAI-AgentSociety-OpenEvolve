import os
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import JSONSearchTool, SerperDevTool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from typing import List
import os

def build_llm() -> LLM:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing.")

        model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
        return LLM(model=f"groq/{model_name}", api_key=api_key)

    if provider == "nvidia":
        api_key = (
            os.getenv("NVIDIA_NIM_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or ""
        ).strip()

        if not api_key:
            raise ValueError("NVIDIA_NIM_API_KEY or NVIDIA_API_KEY is missing.")

        os.environ["NVIDIA_NIM_API_KEY"] = api_key

        model_name = os.getenv("NVIDIA_MODEL_NAME", "meta/llama-3.1-8b-instruct")
        return LLM(
            model=f"nvidia_nim/{model_name}", 
            api_key=api_key,
            max_retries=5,
        )

    if provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL_NAME", "phi3")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return LLM(model=f"ollama/{model_name}", base_url=base_url)

    raise ValueError(
        f"Unsupported LLM_PROVIDER='{provider}'. "
        "Use one of: groq, nvidia, ollama."
    )

llm = build_llm()

# Bypassing OPENAI_API_KEY error
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "NA"

rag_config = {
    "embedding_model": {
        "provider": "sentence-transformer",
        "config": {
            "model_name": "BAAI/bge-small-en-v1.5"
        }
    }
}

def create_rag_tool(json_path: str, collection_name: str, config: dict, name: str, description: str) -> JSONSearchTool:
    from crewai.utilities.paths import db_storage_path
    from crewai_tools.tools.json_search_tool.json_search_tool import FixedJSONSearchToolSchema
    import sqlite3
    import os
    
    collection_exists = False
    db_file = os.path.join(db_storage_path(), "chroma.sqlite3")
    
    if os.path.exists(db_file):
        try:
            # Check native sqlite3 for existing collection to heavily avoid 100% JSON text synchronous chunking bottleneck
            # and avoid ChromaDB singleton initialization conflicts with CrewAI's internal Settings
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
            if cursor.fetchone() is not None:
                collection_exists = True
            conn.close()
        except Exception:
            pass

    if collection_exists:
        print("================COLLECTION EXISTS==================")
        tool = JSONSearchTool(collection_name=collection_name, config=config)
        # CRITICAL: Force the Pydantic schema to hide json_path from the Agent, 
        # so it doesn't trigger validation errors or pass the path and trigger the 3-hour hash loop!
        tool.args_schema = FixedJSONSearchToolSchema
    else:
        print("================COLLECTION NOT EXIST===============")
        tool = JSONSearchTool(json_path=json_path, collection_name=collection_name, config=config)
        
    tool.name = name
    tool.description = description
    return tool

# Preparing RAG Tools
user_rag_tool = create_rag_tool(
    json_path='data/user_subset.json',
    collection_name='v3_hf_user_data',
    config=rag_config,
    name="search_user_profile_data",
    description=(
        "Searches the user profile database using semantic similarity. "
        "This tool accepts ONLY a single plain text string as 'search_query'. "
        "CORRECT: search_query='review habits and average stars for user _BcWyKQL16' "
        "INCORRECT: search_query={'user_id': '_BcWyKQL16', 'query': '...'} "
        "INCORRECT: passing any JSON object, dict, or extra fields. "
        "Always embed the user_id naturally inside the search_query string."
    )
)

item_rag_tool = create_rag_tool(
    json_path='data/item_subset.json',
    collection_name='v3_hf_item_data',
    config=rag_config,
    name="search_restaurant_feature_data",
    description=(
        "Searches the restaurant/business database using semantic similarity. "
        "This tool accepts ONLY a single plain text string as 'search_query'. "
        "CORRECT: search_query='categories location and star rating for business abc123' "
        "INCORRECT: search_query={'item_id': 'abc123', 'query': '...'} "
        "INCORRECT: passing any JSON object, dict, or extra fields. "
        "Always embed the business_id naturally inside the search_query string."
    )
)

review_rag_tool = create_rag_tool(
    json_path='data/review_subset.json',
    collection_name='v3_hf_review_data',
    config=rag_config,
    name="search_historical_reviews_data",
    description=(
        "Searches historical review texts using semantic similarity. "
        "This tool accepts ONLY a single plain text string as 'search_query'. "
        "CORRECT: search_query='past reviews by user _BcWyKQL16 about food quality and service' "
        "INCORRECT: search_query={'user_id': '_BcWyKQL16', 'query': '...'} "
        "INCORRECT: passing any JSON object, dict, or extra fields. "
        "Always embed both user_id and topic naturally inside the search_query string."
    )
)

web_search_tool = SerperDevTool()

# Inject Global Background Knowledge
with open('data/Yelp Data Translation.md', 'r', encoding='utf-8') as f:
    schema_content = f.read()

schema_knowledge = StringKnowledgeSource(
    content=schema_content,
    metadata={"source": "Yelp Schema Definition"}
)

@CrewBase
class SimulationCrew():
    """Yelp Recommendation Crew integrated into SimulationCrew"""
    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = '../../config/agents.yaml'
    tasks_config = '../../config/tasks.yaml'

    @agent
    def user_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['user_analyst'],
            tools=[user_rag_tool, review_rag_tool],
            llm=llm,
            verbose=True,
            max_iter=3,
        )

    @agent
    def item_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['item_analyst'],
            tools=[item_rag_tool, review_rag_tool],
            llm=llm,
            verbose=True,
            max_iter=3,
        )

    @agent
    def web_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["web_researcher"],
            tools=[web_search_tool],
            llm=llm,
            verbose=True,
            max_iter=3,
        )

    @agent
    def eda_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["eda_specialist"],
            tools=[review_rag_tool],
            llm=llm,
            verbose=True,
            max_iter=3,
        )

    @agent
    def prediction_modeler(self) -> Agent:
        return Agent(
            config=self.agents_config['prediction_modeler'], # type: ignore[index]
            llm=llm,
            verbose=True,
            max_iter=3,
        )

    @task
    def analyze_user_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_user_task'],
        )

    @task
    def analyze_item_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_item_task'], # type: ignore[index]
        )

    @task
    def web_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['web_research_task'],
        )

    @task
    def eda_task(self) -> Task:
        return Task(
            config=self.tasks_config["eda_task"],
        )

    @task
    def predict_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['predict_review_task'],
            output_file='report.json'
        )
        
    @crew
    def crew(self) -> Crew:
        """
        Sequential Crew with Cascade Pattern
        """
        return Crew(
            agents=[self.user_analyst(),
                    self.item_analyst(),
                    self.web_researcher(),
                    self.eda_specialist(),
                    self.prediction_modeler()],
            tasks=self.tasks,
            process=Process.sequential,
            knowledge_sources=[schema_knowledge],
            embedder={
                "provider": "huggingface",
                "config": {
                    "model": "BAAI/bge-small-en-v1.5"
                }
            },
            verbose=True,
            max_rpm=10
        )
