# %%
!pip install gradio

# %%
!pip install -q pypdf
!pip install -q python-dotenv

# %%
!pip install -q transformers

# %%
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install  llama-cpp-python --no-cache-dir

# %%
!pip install -q llama-index

# %%
!pip -q install sentence-transformers

# %%
!pip install langchain

# %%
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

from llama_index.core import ServiceContext

# %%
!pip install llama_index

# %%
!mkdir Data
# !wget https://drive.google.com/uc?id=1b5JC1K5TRbYqESSHz-fgftuF1npiQvpA -O Data/Neozep.pdf

!gdown 1gfBNlYCHVufQCPZZG_za7iXE36kOi-iq -O Data/fashion.pdf

# %%
documents = SimpleDirectoryReader("/content/Data/").load_data()

# %%
%pip install llama-index-embeddings-huggingface
%pip install llama-index-llms-llama-cpp

# %%
print(documents)

# %%
import torch
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
#from llama_index.llms import LlamaCPP
#from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

# %%
import llama_index

dir(llama_index.core.indices.service_context.ServiceContext)


# %%
%pip install llama-index-embeddings-langchain

# %%
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from llama_index.embeddings.langchain import LangchainEmbedding
#from llama_index.embeddings import LangchainEmbedding
#from llama_index import ServiceContext
from llama_index.core.indices.service_context import ServiceContext

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="thenlper/gte-large")
)














# %%
service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model
)

# %%


# %%
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# %%
query_engine = index.as_query_engine()

print(query_engine.query("what is the side effects of the neozep"))

# %%
import gradio as gr

def text_to_uppercase(text):
    return query_engine.query(text)

iface = gr.Interface(fn=text_to_uppercase, inputs="text", outputs="text")
iface.launch(debug=True)

# %%



