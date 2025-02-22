import os
from datetime import datetime

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Set API keys (Replace with your actual API keys)
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Web search tool
search_tool = SerperDevTool()

# Researcher Agent: Finds relevant online content
researcher = Agent(
    role="Web Researcher",
    goal="Find the most relevant and engaging content about {topic}",
    verbose=True,
    memory=True,
    backstory=(
        "You are an expert in finding valuable information online."
        "You analyze search results to pick the most useful insights."
    ),
    tools=[search_tool]
)

# Scriptwriter Agent: Creates a catchy YouTube intro script
scriptwriter = Agent(
    role="YouTube Scriptwriter",
    goal="Craft a compelling and engaging intro script for a YouTube video about {topic}.",
    verbose=True,
    memory=True,
    backstory=(
        "You are a skilled storyteller who knows how to hook an audience."
        "Your scripts are engaging, exciting, and always include a strong call to action."
    )
)

# Task 1: Research the given topic
research_task = Task(
    description=(
        "Use online resources to research {topic}."
        "Find the latest, most relevant, and engaging content on the topic."
        "Summarize the key points for the scriptwriter."
    ),
    expected_output="A summary of key insights about {topic}.",
    tools=[search_tool],
    agent=researcher
)

# Task 2: Write the YouTube intro script
scriptwriting_task = Task(
    description=(
        "Based on the research, write a catchy YouTube intro script about {topic}."
        "Decide on the best style (dramatic, humorous, educational, etc.)."
        "Ensure the script grabs the viewer's attention within the first 10 seconds."
        "Include a clear call-to-action at the end."
    ),
    expected_output="A well-written, engaging YouTube intro script stored in 'youtube_intro_script.txt'.",
    agent=scriptwriter,
    output_file=f"youtube_intro_script_{datetime.now()}.txt"
)

# Assemble the Crew
crew = Crew(
    agents=[researcher, scriptwriter],
    tasks=[research_task, scriptwriting_task],
    process=Process.sequential  # The researcher works first, then the scriptwriter
)

# Run the Crew
result = crew.kickoff(
    inputs={"topic": "Explore document processing capabilities with the Gemini API"})
print(result)
