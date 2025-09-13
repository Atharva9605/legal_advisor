import asyncio
import json
from typing import AsyncGenerator, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, AIMessage
from dotenv import load_dotenv
import os

# Import your existing modules
from chains import first_responder_chain, revisor_chain, validator, pydantic_parser
from execute_tools import execute_tools, tavily_tool
from reflexion_graph import app as langgraph_app
from schema import AnswerQuestion, ReviseAnswer

load_dotenv()

# Validate required environment variables
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is required in .env file")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY is required in .env file")

app = FastAPI(title="Legal Advisor AI", description="AI Legal Strategos with Streaming")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateDirectiveRequest(BaseModel):
    case_details: str

class SSEEvent:
    def __init__(self, event_type: str, data: Any):
        self.event_type = event_type
        self.data = data
    
    def format(self) -> str:
        """Format as Server-Sent Event"""
        data_str = json.dumps(self.data) if isinstance(self.data, (dict, list)) else str(self.data)
        return f"event: {self.event_type}\ndata: {data_str}\n\n"

async def stream_draft_response(chain, human_message: HumanMessage) -> AsyncGenerator[SSEEvent, None]:
    """Stream the initial draft response"""
    try:
        # Check if LLM supports streaming
        if hasattr(chain.last, 'astream'):
            # Stream the response in chunks
            full_response = None
            async for chunk in chain.astream([human_message]):
                if hasattr(chunk, 'content') and chunk.content:
                    yield SSEEvent("draft_chunk", {"content": chunk.content})
                full_response = chunk
            
            # Yield the complete response for tool extraction
            if full_response:
                yield SSEEvent("draft_complete", {"response": full_response})
        else:
            # Fallback: invoke normally and yield full response
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: chain.invoke([human_message])
            )
            yield SSEEvent("draft", {"response": response})
    except Exception as e:
        yield SSEEvent("error", {"message": f"Draft generation failed: {str(e)}"})

async def extract_and_stream_reflection(ai_message: AIMessage) -> AsyncGenerator[SSEEvent, None]:
    """Extract and stream reflection from AI message"""
    try:
        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            for tool_call in ai_message.tool_calls:
                if tool_call["name"] == "AnswerQuestion":
                    args = tool_call["args"]
                    
                    # Stream reflection
                    reflection = args.get("reflection", {})
                    yield SSEEvent("reflection", {
                        "missing": reflection.get("missing", ""),
                        "superfluous": reflection.get("superfluous", "")
                    })
                    
                    # Stream search queries
                    search_queries = args.get("search_queries", [])
                    yield SSEEvent("search_queries", {"queries": search_queries})
                    
                    break
    except Exception as e:
        yield SSEEvent("error", {"message": f"Reflection extraction failed: {str(e)}"})

async def stream_search_execution(state: List[BaseMessage]) -> AsyncGenerator[SSEEvent, None]:
    """Stream search execution results"""
    try:
        last_ai_message: AIMessage = state[-1]
        
        if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
            return
        
        for tool_call in last_ai_message.tool_calls:
            if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
                search_queries = tool_call["args"].get("search_queries", [])
                
                for query in search_queries:
                    try:
                        yield SSEEvent("search_executing", {"query": query})
                        
                        # Execute search in thread pool to avoid blocking
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, lambda q=query: tavily_tool.invoke(q)
                        )
                        
                        yield SSEEvent("search_result", {
                            "query": query,
                            "result": result
                        })
                    except Exception as e:
                        yield SSEEvent("search_error", {
                            "query": query,
                            "error": str(e)
                        })
                        
    except Exception as e:
        yield SSEEvent("error", {"message": f"Search execution failed: {str(e)}"})

async def stream_revision_response(chain, state: List[BaseMessage]) -> AsyncGenerator[SSEEvent, None]:
    """Stream the revision response"""
    try:
        # Check if LLM supports streaming
        if hasattr(chain.last, 'astream'):
            # Stream the revision in chunks
            full_response = None
            async for chunk in chain.astream(state):
                if hasattr(chunk, 'content') and chunk.content:
                    yield SSEEvent("revision_chunk", {"content": chunk.content})
                full_response = chunk
            
            # Extract final revised answer
            if full_response and hasattr(full_response, "tool_calls") and full_response.tool_calls:
                for tool_call in full_response.tool_calls:
                    if tool_call["name"] == "ReviseAnswer":
                        args = tool_call["args"]
                        yield SSEEvent("revision_complete", {
                            "answer": args.get("answer", ""),
                            "references": args.get("references", [])
                        })
                        break
        else:
            # Fallback: invoke normally
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: chain.invoke(state)
            )
            yield SSEEvent("revision", {"response": response})
            
    except Exception as e:
        yield SSEEvent("error", {"message": f"Revision failed: {str(e)}"})

async def stream_legal_directive(case_details: str) -> AsyncGenerator[str, None]:
    """Main streaming function that orchestrates the entire process"""
    try:
        human_message = HumanMessage(content=case_details)
        state = [human_message]
        
        yield SSEEvent("start", {"message": "Starting legal directive generation"}).format()
        
        # Step 1: Generate initial draft
        yield SSEEvent("stage", {"current": "draft", "description": "Generating initial draft"}).format()
        
        draft_response = None
        async for event in stream_draft_response(first_responder_chain, human_message):
            yield event.format()
            if event.event_type == "draft_complete":
                draft_response = event.data["response"]
            elif event.event_type == "draft":
                draft_response = event.data["response"]
        
        if not draft_response:
            yield SSEEvent("error", {"message": "Failed to generate draft"}).format()
            return
            
        state.append(draft_response)
        
        # Extract and stream reflection
        async for event in extract_and_stream_reflection(draft_response):
            yield event.format()
        
        # Iteration loop (max 2 iterations)
        iteration = 0
        MAX_ITERATIONS = 2
        
        while iteration < MAX_ITERATIONS:
            iteration += 1
            yield SSEEvent("iteration", {"current": iteration, "max": MAX_ITERATIONS}).format()
            
            # Step 2: Execute search tools
            yield SSEEvent("stage", {"current": "search", "description": f"Executing search queries - Iteration {iteration}"}).format()
            
            async for event in stream_search_execution(state):
                yield event.format()
            
            # Execute tools to get ToolMessage results
            tool_messages = await asyncio.get_event_loop().run_in_executor(
                None, lambda: execute_tools(state)
            )
            
            state.extend(tool_messages)
            
            # Step 3: Generate revision
            yield SSEEvent("stage", {"current": "revision", "description": f"Generating revision - Iteration {iteration}"}).format()
            
            revision_response = None
            async for event in stream_revision_response(revisor_chain, state):
                yield event.format()
                if event.event_type == "revision_complete":
                    revision_response = event.data
                elif event.event_type == "revision":
                    revision_response = event.data["response"]
            
            if revision_response:
                # Check if we have a proper revision response
                if isinstance(revision_response, dict) and "answer" in revision_response:
                    # We got the final answer
                    final_answer = revision_response
                    break
                else:
                    # Add the AI response to state for next iteration
                    state.append(revision_response)
                    
                    # Check if we should continue (based on tool messages count)
                    tool_count = sum(isinstance(msg, ToolMessage) for msg in state)
                    if tool_count > MAX_ITERATIONS:
                        break
            else:
                yield SSEEvent("error", {"message": f"Revision failed in iteration {iteration}"}).format()
                break
        
        # Extract final answer from the last response
        if state:
            last_response = state[-1]
            if hasattr(last_response, "tool_calls") and last_response.tool_calls:
                for tool_call in last_response.tool_calls:
                    if tool_call["name"] in ["ReviseAnswer", "AnswerQuestion"]:
                        final_answer = tool_call["args"].get("answer", "")
                        references = tool_call["args"].get("references", [])
                        
                        yield SSEEvent("final", {
                            "answer": final_answer,
                            "references": references,
                            "iterations": iteration
                        }).format()
                        break
        
        yield SSEEvent("done", {"message": "Legal directive generation completed"}).format()
        
    except Exception as e:
        yield SSEEvent("error", {"message": f"Generation failed: {str(e)}"}).format()

@app.post("/generate_directive")
async def generate_directive(request: GenerateDirectiveRequest):
    """Generate legal directive with streaming response"""
    try:
        return StreamingResponse(
            stream_legal_directive(request.case_details),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Legal Advisor AI is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8003)