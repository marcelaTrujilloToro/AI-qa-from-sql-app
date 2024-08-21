"""QA from SQL App"""

import os
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]


llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


sqlite_db_path = "data/street_tree_db.sqlite"

db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")


chain = create_sql_query_chain(llm, db)

response = chain.invoke(
    {"question": "List the species of trees that are present in San Francisco"})

print("\n----------\n")

print("List the species of trees that are present in San Francisco")

print("\n----------\n")
print(response)

print("\n----------\n")

print("Query executed:")

print("\n----------\n")

print(db.run(response))

print("\n----------\n")


execute_query = QuerySQLDataBaseTool(db=db)

write_query = create_sql_query_chain(llm, db)

chain = write_query | execute_query

response = chain.invoke(
    {"question": "List the species of trees that are present in San Francisco"})

print("\n----------\n")

print("List the species of trees that are present in San Francisco (with query execution included)")

print("\n----------\n")
print(response)

print("\n----------\n")


answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke(
    {"question": "List the species of trees that are present in San Francisco"})

print("\n----------\n")

print("List the species of trees that are present in San Francisco (passing question and result to the LLM)")

print("\n----------\n")
print(response)

print("\n----------\n")
