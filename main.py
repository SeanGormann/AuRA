import os
#from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
#from fastapi import FastAPI
import streamlit as st

#load_dotenv()
brwoserless_api_key = os.environ.get("BROWSERLESSAPIKEY")
serper_api_key = os.environ.get("SERPAPIKEY")
openai_api_key = os.environ.get("OPENAIAPIKEY")

# Search Tool
from serpapi import GoogleSearch
from datetime import datetime


#### Tool for searching
def search(combined_input, starting_from=0):
    # Split the combined input by comma
    parts = combined_input.split(", ")
    
    # If there's only one part, it's just the query
    if len(parts) == 1:
        query = parts[0]
        starting_from = 0
    else:
        # Extract the query and starting_from values
        query = ", ".join(parts[:-1])  # This takes all parts of the split except the last one
        try:
            starting_from = int(parts[-1].strip())  # Convert the last part to integer
            if not (0 <= starting_from <= 20):
                raise ValueError("Value out of range")
        except ValueError:
            print(f"Invalid 'starting_from' value: {parts[-1]}")
            return {"error": f"Invalid 'starting_from' value: {parts[-1]}", "output": f"Error: Invalid 'starting_from' value: {parts[-1]}"}

    print(f"Searching for '{query}' starting from {starting_from}")
    try:
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": serper_api_key
        }

        # Check if starting_from has a value and set the asylo parameter accordingly
        if starting_from > 0:
            current_year = datetime.now().year
            params["asylo"] = str(current_year - starting_from)

        # If you want articles from the last year sorted by date, you can use the scisbd parameter
        # params["scisbd"] = "1"  # This will include only articles from the last year sorted by date
        search_instance = GoogleSearch(params)
        results = search_instance.get_dict()
        
        return results #formatted_results

    except Exception as e:
        print(f"Error encountered while searching: {e}")
        return {"error": f"Failed to fetch search results: {str(e)}", "output": f"Error: {str(e)}"}



#### Tool for summarizing content
def summary(objective, content):
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    # Check if the output exceeds 16k tokens and truncate if necessary
    max_tokens = 16000
    if len(output["choices"]["text"].split()) > max_tokens:
        output["choices"]["text"] = ' '.join(output["choices"]["text"].split()[:max_tokens])

    return output




#### Tool for scraping website
def scrape_website(objective: str, url: str):
    print("Scraping website...")

    # Define request headers 
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent 
    data = {
        "url": url
    }

    # Python --> JSON 
    data_json = json.dumps(data)

    # Send POST request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Error handling
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]



#### Modify the system message to include the guidelines, reduce hallucinations and provide sources
system_message = SystemMessage(
    content="""Welcome, esteemed researcher! Your expertise spans the realms of scientific inquiry, enabling you to delve into intricate research across various domains and yield empirically grounded results. Your commitment is unwavering: you meticulously gather and analyze data to underpin your investigations.

    To fulfill your mission, kindly adhere to the following guidelines:
    1/ Conduct thorough research to amass an exhaustive compendium of information pertinent to the objective.
    2/ When relevant, extract insights from URLs, articles, and external sources to bolster the depth of your exploration.
    3/ Ponder if additional searches and extractions would elevate the caliber of your research. If affirmative, proceed. However, restrict this iteration to no more than three cycles.
    4/ Your contributions must remain grounded in verifiable facts and substantiated data—eschew fabrication at all costs.
    5/ In your final presentation, provide an exhaustive catalog of references and links that corroborate and augment your findings.
    6/ Reiterate: ensure that your final deliverable meticulously cites all sources, bolstering your research's credibility and validity."""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}


##### Define Agent
llm = ChatOpenAI(openai_api_key = openai_api_key, temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)



#### Function for calling the agent
def call_agent(query, starting_from):
    combined_input = f"{query}, {str(starting_from)}"
    print(f"Combined Input: {combined_input}")
    return agent({"input": combined_input})






# ----------------------------
# Use streamlit to create a web app
import streamlit as st

st.set_page_config(page_title="Aura", page_icon=":scientist:")

def main():
    # Sidebar menu
    st.sidebar.title("Menu")
    page_selection = st.sidebar.selectbox("Choose Page", ["AuRA", "Welcome!"])

    # Add a slider to the sidebar for selecting years to look back
    st.sidebar.write("How far do you want me to look back?")
    years_ago = st.sidebar.slider("Years ago", 0, 20, 0)  # Slider from 0 to 20 with default value set to 0

    if page_selection == "Welcome!":
        # Display Landing Page content here
        st.title("Hi, I'm Aura! :scientist: Your very own Autonomous Research Assistant :fire:")
        st.markdown("### AutoGPT applied to Research")
        
        #st.image("your_image_url_here", use_column_width=True)  # Add an image to enhance the visual appeal
        
        st.write("Welcome to Aura, your dedicated research companion. I'm here to empower researchers at all levels and assist you in staying at the forefront of cutting-edge research in any domain. I'll sift through Google Scholar to uncover recent, pertinent papers on your chosen topic and offer concise summaries. If you're eager for more, I'll equip you with a trove of reference links for deeper exploration.")
        
        st.markdown("### Getting Started is a Breeze")
        st.write("1. **Define Your Research Goal:** Enter your research query in the textbox below. Whether it's a broad inquiry like 'What is ALS?' or a focused question about 'GLT25D1's role in human collagen production,' I'm here to help.")
        st.write("2. **Adjust the Timeframe:** Utilize the year slider to determine the research timeframe. Slide left to include more recent findings or leave it at 0 to encompass all years.")
        st.write("3. **Navigate to AuRA:** Open the dropdown menu on the left and select the 'AuRA' tab to embark on your research journey.")
        st.write("4. **Weekly Reports (Coming Soon...):** Enter your email and I'll send you weekly summaries on your desired research interest .")

        
        st.markdown("#### Let's Explore the World of Knowledge Together!")


    elif page_selection == "AuRA":
        st.header("AuRA :scientist:")
        query = st.text_input("What are we researching today? Let's find out just how deep the rabbit hole goes...")

        if query:
            st.write("Investigating ", query)
            
            # This is a placeholder, replace with your agent call
            result = call_agent(query, years_ago)

            st.info(result['output'])

    # Info Section in Sidebar
    st.sidebar.subheader("Info")
    st.sidebar.write("I'm an AutoGPT that uses SOTA models to analyse papers published in google scholar on your research interest. I am by no means a substitute for *you* reading scientific papers. I merely hope to help you sift through the noise and find the papers that are most relevant to your research.")

    # Contact Section in Sidebar
    st.sidebar.subheader("Contact")
    st.sidebar.write("Email: seangorman117@gmail.com")
    st.sidebar.write("GitHub: https://github.com/SeanGormann")

if __name__ == '__main__':
    main()
