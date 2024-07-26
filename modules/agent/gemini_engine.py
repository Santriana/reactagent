from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    PromptTemplate,
)
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
import os
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import sqlalchemy as db
import cohere
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent import (
    RunnableAgent,
    AgentExecutor
)
from langchain.agents import create_react_agent
# from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain.agents.agent import AgentOutputParser
import nest_asyncio
import asyncio
from dotenv import load_dotenv

load_dotenv('.env')


username = 'YOUR_DB_USERNAME'
password = 'YOUR_DB_PASSWORD'
hostname = 'YOUR_DB_HOSTNAME'
port = 'YOUR_DB_PORT'
namespace = 'YOUR_NAMESPACE'
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"


TEMPLATE_1 = """You are an expert in create sql query.
You are working with a {dialect} database. The tables that you can use is {db_name}.
You should remember that user always want to know the answer in detail and always use the data from the database that you have.

You should use the tools below to answer the question posed of you:
{tools}

Please follow this instructions:
<instructions>
- Please strictly answer the question, never create your own question
- Please strictly use the data from tables {db_name} to create your reason
- Please never put any information after "Action:" and "Action Input:" except the tool name
- Please NEVER MAKE ANY ASSUMPTION that not related with the data
- Please always try to understand the question in your plan
- Please always give the reason why you give the final answer, user always corious about the reason and never give the answer like etc, many more, or need more analyst
- You only can run SELECT query to answer the question and also INSERT to table DESCRIPT and EXAMPLE ONLY, you cannot run UPDATE, DELETE, or any other query, included show table informations
- Please never answer the question that relate with table schema, table informations, or any other question that not related with the data
- When Asked about column related to "Name" for example ("OrganizationName", "CompanyName", "Address", etc.) ALWAYS USE syntax LIKE [%].
- For WHY question, you can answer directly without querying, you can use information from result expectation column to answer the question
</instructions>


#####
Before create the query, please follow this steps:
- Step 1: Understand the questions
- Step 2: You must understand the question and the example query from provided example {get_example}
- Step 3: Plan the select column from the question
- Step 4: IF NEEDED, Plan the ordering, grouping, and joining statement from the question
- Step 5: Thought about the plan
- Step 6: You know how to combine the query
- Step 7: Combine all your plan and return the query
#####


Please use the following format to answer the question:
<output_format>
Question: the input question you must answer
Plan: you should understand the question first before plan what to do
Thought: you should always think about what to do according to the plan
Action: the action to take, can be one of [{tool_names}]
Action Input: what to instruct the AI Action representative
Observation: the result of the action input
... (IF NEEDED this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question, don't use any abbreviations or ellipsis.

Beside that, the output from parser SHOULD NOT INCLUDE this Character ("```sql", "```", "## ", "**")

</output_format>
When responding with your Final Answer, remember that the person you are responding to CANNOT see any of your Thought/Action/Action Input/Observations, so if there is any relevant information there you need to include it explicitly in your response.

The description of columns in the tables are:
{get_descript}

Please only use the column that listed above


Reflection:
- Top 10 doesn't mean you have to limit 10 data, you can limit 100 data if you need to filter the data
- Always use the data from the database that you have
- `why` and `how` question always need the reason from the Comment column
- The worst means the lowest value from the data that asked


Begin!
Question: {input}
Thought: {agent_scratchpad}."""

FINAL_ANSWER_ACTION = "Final Answer:"


class CustomOutputParser(AgentOutputParser):
    def parse(self, text: str):
        includes_answer = FINAL_ANSWER_ACTION in text
        singleOutput = ReActSingleInputOutputParser()
        jsonOutput = ReActJsonSingleInputOutputParser()
        text = text.replace('```sql', '').replace('```', '').replace('## ','').replace('**','').replace('- Action Input: ', 'Action Input: ').replace('`', '')
        return singleOutput.parse(text) if not includes_answer else jsonOutput.parse(text)

nest_asyncio.apply()

class queryreactagent:
    # def __init__(self):
    #     self.llm = AzureChatOpenAI(deployment_name = "gpt-4o",
    #                      api_key = "1bc94464956c4a4cb6f7430de96be615",
    #                      openai_api_version="2024-02-01",
    #                      azure_endpoint="https://iris-openai-azure.openai.azure.com/",
    #                      temperature=0)
    #     self.emb_key = os.getenv('COHERE_API_KEY')
    #     if not self.emb_key:
    #         raise ValueError("COHERE_API_KEY environment variable not set")
        
    #     self.db, self.connection = self.setupDB()
    #     self.similar_list_ex = []
    #     self.similar_list_des = []

    def __init__(self):
        self.llm = self.setup_llm()
        
        self.emb_key = os.getenv('COHERE_API_KEY')
        if not self.emb_key:
            raise ValueError("COHERE_API_KEY environment variable not set")
        
        self.db, self.connection = self.setupDB()
        self.similar_list_ex = []
        self.similar_list_des = []
    
    def setup_llm(self):
        loop = asyncio.new_event_loop() 
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.init_llm_async())

    async def init_llm_async(self):
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key='YOUR_GOOGLE_API_KEY',
            temperature=0
        )

    def setupDB(self):
        engine = create_engine(CONNECTION_STRING)
        db = SQLDatabase(engine=engine, max_string_length=0)
        connection = engine.connect()
        return db, connection
    
    def get_prompt(self, dialect, db_name, get_descript, get_example):
        prompt = PromptTemplate.from_template(TEMPLATE_1)
        partial_prompt = prompt.partial()

        partial_prompt = partial_prompt.partial(
            dialect = dialect,
            db_name = db_name,
            get_descript = get_descript,
            get_example = get_example
        )
        return partial_prompt

    def get_top_similar(self, query):
        df_ex = pd.read_sql_table("example_query", self.connection)
        df_ex['embeddings'] = df_ex['embeddings'].apply(lambda x: np.array(json.loads(x)))

        df_des = pd.read_sql_table("description_tables", self.connection)
        df_des['embeddings'] = df_des['embeddings'].apply(lambda x: np.array(json.loads(x)))

        co = cohere.Client(self.emb_key)
        query_embedding = co.embed(texts=[query], model='embed-english-v2.0').embeddings[0]

        # Similarity searchs for examples
        embeddings_ex = np.vstack(df_ex['embeddings'].values)
        similarity_scores_ex = cosine_similarity([query_embedding], embeddings_ex)
        top_3_ex = np.argsort(similarity_scores_ex[0])[-3:][::-1]
        top_3_ex = df_ex.iloc[top_3_ex]

        # Similarity searchs for descriptions
        embeddings_des = np.vstack(df_des['embeddings'].values)
        similarity_scores_des = cosine_similarity([query_embedding], embeddings_des)
        top_3_des = np.argsort(similarity_scores_des[0])[-3:][::-1]
        top_3_des = df_des.iloc[top_3_des]

        # similar_list_ex = []
        for index, row in top_3_ex.iterrows():
            self.similar_list_ex.append({
                "question": row['Question'],
                "query": row['sql query']
            })

        # similar_list_des = []
        for index, row in top_3_des.iterrows():
            self.similar_list_des.append({
                "description": row['description_table'],
                "column_name": row['column_name']
            })
        
        self.connection.close()
        
        return self.similar_list_ex, self.similar_list_des

    def setAgent(self):
        toolkit = SQLDatabaseToolkit(llm=self.llm, db=self.db)
        tools = toolkit.get_tools()

        prompt = self.get_prompt(
            self.db.dialect, 
            self.db.get_usable_table_names(),
            self.similar_list_des,
            self.similar_list_ex
        )
        agent = RunnableAgent(
            runnable=create_react_agent(self.llm, tools, prompt, output_parser = CustomOutputParser()),
            input_keys_arg=["input"],
            return_keys_arg=["output"],
        )
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=False,
            max_iterations=5,
            early_stopping_method='force',
            handle_parsing_errors=True,
        )

    def ask(self, question):
        self.get_top_similar(query=question)
        agent_executor = self.setAgent()
        res = agent_executor.invoke({"input": question})
        return res