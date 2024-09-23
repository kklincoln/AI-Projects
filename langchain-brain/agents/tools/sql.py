import sqlite3 
from pydantic.v1 import BaseModel # this is a library that allows us to annotate different classes inside a python class to clearly define the data that we expect it to receive as attributes
from typing import List
from langchain.tools import Tool #allows langchain to create tools for calling specific actions

conn = sqlite3.connect("db.sqlite")

# function to list all tables in the database
def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return "\n".join(row[0] for row in rows if row[0] is not None) # return a tablename per carriage return, only if the table name is not None

# function that should be executed whenever CGPT decides it needs to execute a SQLite query
def run_sqlite_query(query):
    c = conn.cursor() #stablish a cursor to access and navigate through the db
    try:
        c.execute(query) #execute the CGPT created query that was passed as an argument during function call
        results = c.fetchall() #return all of the rows in results of the query and sends back to cGPT
        c.close() #close the cursor to free up resources
        return results  # Return the fetched results
    except sqlite3.OperationalError as err:
        c.close() #close the cursor to free up resources
        return f"An error occurred: {str(err)}" 




#creates a record that says if you want to be a class of RunQueryArgsSchema, you must provide a 'query' attr that is a string. 
class RunQueryArgsSchema(BaseModel):
    query: str

#create the tool from the toolbelt to run queries
run_query_tool = Tool.from_function(
    name = "run_sqlite_query",
    description="Run a sqlite query.",
    func=run_sqlite_query, #the function that should be called when CGPT decides to use this tool (the one that executes queries)
    args_schema = RunQueryArgsSchema # langchain uses this to better describe the arguments that CGPT should provide to our tools
)



# create a function and a tool associated with describing the tables
def describe_tables(table_names):
    c = conn.cursor()
    # "'users','orders','products'"
    tables= ', '.join("'" + item + "'" for item in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name IN ({tables})")
    return '\n'.join(row[0] for row in rows if row[0] is not None) #for every table, create a new line in the output that describes the table from the sqlite_master table

# establish the expectation that for the describe_tables function to work, an argument should exist for table_names which is a List of strings.
class DescribeTablesArgsSchema(BaseModel):
    table_names: List[str]

describe_tables_tool = Tool.from_function(
    name="describe_tables",
    description="Given a list of table names, returns the schema of those tables from sqlite_master.",
    func=describe_tables,
    args_schema = DescribeTablesArgsSchema
)