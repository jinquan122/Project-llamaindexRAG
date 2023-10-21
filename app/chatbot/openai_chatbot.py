from llama_index import ServiceContext, PromptHelper
from llama_index import get_response_synthesizer
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import OpenAIAgent
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from app.prompt import text_qa_template
from llama_index.llms import OpenAI


class openai_gpt:
    def llamaindexAgent(
            self, 
            index, 
            context_window: int = 4096, 
            num_output: int = 256, 
            chunk_overlap_ratio: float = 0.1, 
            similarity_top_k: int = 10,
            model: str = "gpt-3.5-turbo-0613",
            temperature = 0) -> callable:
        '''
        To answer question based on database knowledge.

        :Params: index = vector database index form llamaindex (pinecone as the vector database)

        Parties: 
        1. Agent = OpenAI Agent
        2. LLM = OpenAI gpt-3.5-turbo-0613
        '''
        
        prompt_helper = PromptHelper(
            context_window = context_window, 
            num_output = num_output, 
            chunk_overlap_ratio = chunk_overlap_ratio, 
            chunk_size_limit = None
            )
        
        service_context = ServiceContext.from_defaults(
            prompt_helper = prompt_helper
            )
        
        retriever = VectorIndexRetriever(
            index = index, 
            similarity_top_k = similarity_top_k,
        )

        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=text_qa_template,
            service_context=service_context
        )

        vector_db_query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer
        )

        vector_query_engine_tool = QueryEngineTool(
            query_engine = vector_db_query_engine,
            metadata = ToolMetadata(
            name="your_database_name",
            description="your_description",
            )
        )

        return OpenAIAgent.from_tools(
            [vector_query_engine_tool],
            llm=OpenAI(model = model, temperature = temperature),
            verbose = False,
            system_prompt=
            "You must follow the rules below: \n"
            "1. your_rules \n"
        )





