from langchain.tools import StructuredTool #allows for calling functions that receive more than one argument; Tool only allows for one
from pydantic.v1 import BaseModel #establish expectations for the datatypes associated with the arguments being passed into functions


def write_report(filename, html):
    #open the filename that is provided and write the html output from CGPT to display the report
    with open(filename, "w") as f:
        f.write(html)

#establish expectations for the datatypes associated with the arguments being passed into the write_report function
class WriteReportArgsSchema(BaseModel):
    filename: str
    html: str

#because of legacy decisions, when you create a tool out of the Tool class, it can only use functions that receive a single argument. This one allows multiple args
write_report_tool = StructuredTool(
    name="write_report",
    description="Write an HTML file to disk. Use this tool whenever someone asks for a report.", #tell CGPT how/when this tool is useful
    func=write_report, #what function does this apply to
    args_schema=WriteReportArgsSchema
)