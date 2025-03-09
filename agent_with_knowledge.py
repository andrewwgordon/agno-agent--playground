import sys
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You are an expert on the maintenance of a Diamond DA 42 aircraft",
    instructions=[
        "Search your knowledge base for information on the maintenance of a Diamond DA 42 aircraft.",
        "If the question is better suited for the web, search the web to fill in gaps.",
        "Prefer the information in your knowledge base over the web results."
    ],
    knowledge=PDFUrlKnowledgeBase(
        urls=["http://support.diamond-air.at/fileadmin/uploads/files/after_sales_support/DA42_New_Generation/Airplane_Maintenance_Manual/Basic_Manual/70215-DA42-NG-AMM-r5.pdf"],
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="diamond-da-42",
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small"),
        ),
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=False,
    markdown=True
)

# Comment out after the knowledge base is loaded
# if agent.knowledge is not None:
#     agent.knowledge.load()

while True:
    query = input("\n> ")
    if query == "quit":
        sys.exit(0)
    response = agent.run(query)
    print(response.content)