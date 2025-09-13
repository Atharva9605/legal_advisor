from typing import List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools

load_dotenv()

graph = MessageGraph()
MAX_ITERATIONS = 2

graph.add_node("draft", first_responder_chain)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revisor_chain)


graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "revisor")

def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

graph.add_conditional_edges("revisor", event_loop)
graph.set_entry_point("draft")

app = graph.compile()

print(app.get_graph().draw_mermaid())

response = app.invoke(
    "A Dummy Case Example: The Case of the Faulty Foundation and the Supply Chain\n\nParties Involved\n\nPlaintiff: \"Ar. Rohan Varma & Associates,\" " \
    "a renowned architectural and design firm based in Bangalore.\n\nDefendant 1: The \"Karnataka State Infrastructure Development Corporation (KSIDC),\" a government agency.\n\nDefendant 2: \"Apex Materials Pvt. Ltd.,\"" \
    " a supplier of construction materials.\n\nThe Facts of the Case\n\nIn 2024, Ar. Rohan Varma & Associates was awarded a contract to design and supervise the construction of a new public library in Bangalore. The fixed-price," \
    " \"turnkey\" agreement required the firm to manage all aspects of the project, including the selection of sub-contractors and material suppliers.\n\nDuring the foundation work, being executed by a sub-contractor, significant structural flaws appeared. A third-party geotechnical survey confirmed the foundation was improperly laid and at high risk of failure. This led the KSIDC to issue a stop-work order and subsequently terminate the contract with the architectural firm.\n\nThe situation was further complicated by two new developments:\n\n1. Defective Materials: A forensic analysis of the foundation material revealed that the cement supplied by Apex Materials Pvt. Ltd. was substandard and did not meet the quality specifications outlined in the contract. Ar. Rohan Varma & Associates had chosen Apex Materials from a list of \"preferred vendors\" provided by the KSIDC, although the contract did not make their use mandatory.\n\n2. Unforeseen Delays: A local community group began a series of protests at the construction site, demanding a change to the library's design to include a community garden. These protests, which led to multiple temporary injunctions and work stoppages, were entirely unforeseen by both parties.\n\nThe KSIDC is now suing Ar. Rohan Varma & Associates for breach of contract, negligence, and substantial project delays and cost overruns. Ar. Rohan Varma & Associates, in turn, has filed a counter-suit against the KSIDC for wrongful termination and is also suing Apex Materials Pvt. Ltd. for the financial and reputational damage caused by the defective materials.\n\nThe Legal Issues at Play\n\nThis case, now before the Karnataka High Court, is a complex legal battle with multiple intertwined problems:\n\n1. Liability for Substandard Work: Who is ultimately responsible for the faulty foundation? The architect, for supervising the work; the sub-contractor, for the execution; or the material supplier, for providing defective cement? The court must determine if the architect's contract with the sub-contractor adequately transferred liability, and if the contract with Apex Materials had any clauses for quality assurance and indemnification.\n\n2. Supplier vs. Architect Liability: The case will test the principle of product liability within a construction project. Did the architect have a duty to test the materials provided by a \"preferred vendor,\" or could they reasonably trust the quality of the product? The court must interpret the \"preferred vendor\" clause to determine if it absolved the architect of some responsibility.\n\n3. Force Majeure and Unforeseen Delays: The contract contained a standard \"force majeure\" clause, which typically covers events beyond the control of the parties. The court will need to decide if the local community protests and subsequent work stoppages qualify as a force majeure event, and if so, how the associated delays and costs should be allocated between the parties.\n\n4. Risk Allocation in Public Contracts: This case will set a major precedent on how risk is allocated in public procurement. It will address who bears the burden when multiple factors—poor workmanship, defective materials, and third-party interference—contribute to project failure. The court's ruling will determine the extent of a government agency's liability for recommending vendors and the level of responsibility contractors assume for every aspect of a project."
)


print(response[-1].tool_calls[0]["args"]["answer"])
print(response, "response")