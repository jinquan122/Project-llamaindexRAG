from llama_index import PromptTemplate

text_qa_template = PromptTemplate("""\
"Matching article information is below.\n"
"---------------------\n"
"{context_str}\n"
"---------------------\n"
"Given the above information about the articles and not prior knowledge, "
"answer the question: {query_str}\n"
""")