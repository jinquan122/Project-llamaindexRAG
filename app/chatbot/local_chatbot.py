from torch import cuda, bfloat16
import transformers
from transformers import pipeline
from llama_index.llms.huggingface import HuggingFaceLLM 
from llama_index import ServiceContext, PromptHelper
from llama_index import get_response_synthesizer
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.agent import ReActAgent
from app.prompt import text_qa_template
from llama_index.llms import LlamaCPP
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

class Llama2_7b:
    def __init__(self, local_llm: bool = False, llamacpp: bool = True) -> None:
        
        if local_llm:
            '''
            To load the local LLM (Llama2-7b) using 4 bit quantitization method.
            Prerequisite: 
            1. Llama 2 license authorization from Huggingface.
            '''
            self.model_id = 'meta-llama/Llama-2-7b-chat-hf'
            self.device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type = 'nf4',
                bnb_4bit_use_double_quant = True,
                bnb_4bit_compute_dtype = bfloat16
            )

            hf_auth = config.get('Huggingface', 'auth')
            model_config = transformers.AutoConfig.from_pretrained(
                self.model_id,
                use_auth_token = hf_auth
            )

            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code = True,
                config = model_config,
                quantization_config = bnb_config,
                device_map = self.device,
                use_auth_token = hf_auth
            )

            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            model.eval()

            self.model = model
            self.tokenizer = tokenizer
            self.llm = HuggingFaceLLM(model = self.model, tokenizer = self.tokenizer)

        if llamacpp:
            '''
            To load llamaindex default LLM (Llama2-13b-GGUF)
            '''
            model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

            self.llm = LlamaCPP(
                model_url = model_url,
                model_path = None,
                temperature = 0.1,
                max_new_tokens = 256,
                context_window = 3900,
                generate_kwargs = {},
                model_kwargs = {"n_gpu_layers": 1},
                verbose = False,
            )


    def chat(self, input: dict) -> str:
        '''
        To chat with the chatbot: Chatbot will give response to the user adhering to the input context.

        :Params: input = User chat input which in dict form. Retrieve the input using 'msg' key.

        - The output is the answer to the user input in string form.
        '''

        user_input = input.get('input', None)
        prompt = f'''
        You are given a conversation. You are a friendly and helpful chat agent.\n
        Must chat in singlish style.\n
        User: {user_input}\n
        Chat Agent: 
        '''
        
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        generated_text = pipe(
            prompt, 
            max_length = 500,
            early_stopping = True,
            no_repeat_ngram_size = 2)[0]

        return generated_text['generated_text']
    
    def llamaindexAgent(
            self, 
            index, 
            context_window: int = 4096, 
            num_output: int = 256, 
            chunk_overlap_ratio: float = 0.1, 
            similarity_top_k: int = 10,
            verbose: bool = False,
            max_iterations: int = 3) -> callable:
        '''
        To answer question based on database knowledge.

        :Params: index = vector database index form llamaindex (pinecone as the vector database)

        Parties: 
        1. Agent = React Agent
        2. LLM = Llama2-7b or Llama2-13b based on the __init__ LLM
        '''
        
        prompt_helper = PromptHelper(
            context_window = context_window, 
            num_output = num_output, 
            chunk_overlap_ratio = chunk_overlap_ratio, 
            chunk_size_limit = None
            )
        
        service_context = ServiceContext.from_defaults(
            llm = self.llm,
            prompt_helper = prompt_helper
            )
        
        retriever = VectorIndexRetriever(
            index = index, 
            similarity_top_k = similarity_top_k,
        )

        response_synthesizer = get_response_synthesizer(
            response_mode = "compact",
            text_qa_template = text_qa_template,
            service_context = service_context
        )

        vector_db_query_engine = RetrieverQueryEngine(
            retriever = retriever,
            response_synthesizer = response_synthesizer
        )

        vector_query_engine_tool = QueryEngineTool(
            query_engine = vector_db_query_engine,
            metadata = ToolMetadata(
            name="your_database_name",
            description="your_description",
            )
        )

        return ReActAgent.from_tools(
            [vector_query_engine_tool],
            llm = self.llm,
            verbose = verbose,
            max_iterations = max_iterations
        )






