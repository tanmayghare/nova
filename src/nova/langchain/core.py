from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool

from .config import LangChainConfig

# Define input schemas for tools
class QueryInput(BaseModel):
    question: str = Field(description="The question to ask the RAG system.")

class WebSearchInput(BaseModel):
    query: str = Field(description="The query for web search.")

class AddDocumentsInput(BaseModel):
    documents: List[str] = Field(description="List of text documents to add.")

class AddWebInput(BaseModel):
    urls: List[str] = Field(description="List of URLs to scrape and add content from.")

class NovaLangChain:
    """Core LangChain integration for Nova.
    Provides RAG, web search, and document management capabilities as tools.
    """
    
    def __init__(self, config: Optional[LangChainConfig] = None):
        """Initialize NovaLangChain with configuration."""
        self.config = config or LangChainConfig()
        self.llm = self.config.get_llm()
        self.embeddings = self.config.get_embeddings()
        # Handle potential None vector_store from config
        try:
            self.vector_store = self.config.get_vector_store(self.embeddings)
            if self.vector_store is None:
                print("Warning: Vector store could not be loaded. RAG and document adding will likely fail.")
                self.qa_chain = None # Indicate RAG is unavailable
            else:
                 self.qa_chain = self._create_qa_chain() # Initialize RAG chain only if vector store exists
        except Exception as e:
            print(f"Error initializing vector store or QA chain: {e}. RAG features may be disabled.")
            self.vector_store = None
            self.qa_chain = None
            
        self.search_tool = DuckDuckGoSearchRun() if self.config.web_search_enabled else None
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.rag_chunk_size,
            chunk_overlap=self.config.rag_chunk_overlap
        )
        
        # # Initialize RAG chain - moved to conditional init above
        # self.qa_chain = self._create_qa_chain()
    
    def _create_qa_chain(self) -> Optional[RetrievalQA]:
        """Create the RAG QA chain. Returns None if vector store is unavailable."""
        if not self.vector_store:
             return None
             
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}

        Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    @tool(args_schema=QueryInput)
    async def query_rag(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with a question to get an answer based on stored documents."""
        if not self.qa_chain:
            return {"error": "RAG system not available (Vector Store failed to load)."}
        try:
            # Langchain QA chain expects a dictionary input
            result = await self.qa_chain.ainvoke({"query": question}) 
            return {
                "answer": result.get("result", "No answer found."),
                "sources": [doc.metadata for doc in result.get("source_documents", [])]
            }
        except Exception as e:
             print(f"Error during RAG query: {e}")
             return {"error": f"Error during RAG query: {e}"}

    @tool(args_schema=WebSearchInput)
    async def web_search(self, query: str) -> List[Dict[str, str]]:
        """Perform web search using DuckDuckGo to find recent information."""
        if not self.search_tool:
            return [{"error": "Web search is disabled in configuration."}]
        try:
            # DuckDuckGoSearchRun is synchronous, run in executor?
            # For now, assume it's okay, or replace with an async tool if needed.
            results = self.search_tool.run(query)
            # Limit results and format
            search_results = results.split("\n")[:self.config.web_search_max_results]
            return [{"content": res} for res in search_results if res] # Filter empty results
        except Exception as e:
             print(f"Error during web search: {e}")
             return [{"error": f"Error during web search: {e}"}]

    @tool(args_schema=AddDocumentsInput)
    async def add_documents(self, documents: List[str]) -> Dict[str, str]:
        """Add text documents to the RAG vector store."""
        if not self.vector_store:
            return {"error": "Cannot add documents, vector store not available."}
        try:
            docs = self.text_splitter.create_documents(documents)
            await self.vector_store.aadd_documents(docs)
            # Save is sync, consider executor if it becomes slow
            self.vector_store.save_local(self.config.vector_store_path)
            return {"status": "success", "message": f"Added {len(docs)} document chunks."}
        except Exception as e:
            print(f"Error adding documents: {e}")
            return {"error": f"Error adding documents: {e}"}
    
    @tool(args_schema=AddWebInput)
    async def add_web_content(self, urls: List[str]) -> Dict[str, str]:
        """Scrape content from web page URLs and add it to the RAG vector store."""
        if not self.vector_store:
            return {"error": "Cannot add web content, vector store not available."}
        added_chunks_count = 0
        errors = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs = await loader.aload() # Use async loading
                split_docs = self.text_splitter.split_documents(docs)
                if split_docs:
                    await self.vector_store.aadd_documents(split_docs)
                    added_chunks_count += len(split_docs)
            except Exception as e:
                print(f"Error adding web content from {url}: {e}")
                errors.append(f"Failed to process {url}: {e}")
        
        if added_chunks_count > 0:
             try:
                 # Save is sync, consider executor
                 self.vector_store.save_local(self.config.vector_store_path)
                 message = f"Added {added_chunks_count} document chunks from {len(urls) - len(errors)} URLs."
             except Exception as save_e:
                  message = f"Added {added_chunks_count} chunks, but failed to save vector store: {save_e}"
                  errors.append(f"Failed to save vector store: {save_e}")
        else:
            message = "No new content added."

        return {
            "status": "partial_success" if errors and added_chunks_count > 0 else ("success" if added_chunks_count > 0 else "failure"),
            "message": message,
            "errors": errors
        }
    
    def get_tools(self) -> List[Any]:
        """Return a list of available tools for the agent."""
        tools_list = []
        if self.qa_chain:
            tools_list.append(self.query_rag)
        if self.search_tool:
            tools_list.append(self.web_search)
        if self.vector_store:
            tools_list.append(self.add_documents)
            tools_list.append(self.add_web_content)
        return tools_list 