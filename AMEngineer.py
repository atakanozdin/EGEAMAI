import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import messages_to_dict, messages_from_dict, Document
from langchain.document_transformers.openai_functions import create_metadata_tagger
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.vectorstores import Chroma
import chromadb
from operator import itemgetter
import pandas as pd

os.environ["OPENAI_API_KEY"] = "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"

llm = OpenAI(temperature =0.4)
embedding_function = OpenAIEmbeddings()


# get vector store
def call_vectordb(embedding_function):
    collection_name_1 = input("What is the collection name: ")
    vector_db = Chroma(persist_directory=f"./chroma_db/{collection_name_1}", embedding_function=embedding_function)
    
    retriever = vector_db.as_retriever(lambda_val=0.025, k=7, filter=None)
    return retriever


def prompt_templates():
    #CONDENSE_QUESTION_PROMPT
    _template = """
    [INST]
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language, that can be used to query a FAISS index. This query will be used to retrieve documents with additional context.

    Let me share a couple examples that will be important.

    If you do not see any chat history, you MUST return the "Follow Up Input" as is:

    ```
    Chat History:

    Follow Up Input: What features can I add as an input of the machine learning model for the additive manufacturing balling situations?
    Standalone Question:
    What features can I add as an input of the machine learning model for the additive manufacturing balling situations?
    ```

    If this is the second question onwards, you should properly rephrase the question like this:

    ```
    Chat History:
    Human: What features can I add as an input of the machine learning model for the additive manufacturing balling situations?
    AI:
    There are several features that can be added as inputs to a machine learning model for predicting balling situations in additive manufacturing. Some of these features include:

    1. Process parameters: These include variables such as feed rate, gas flow rate, and power output, which can affect the formation of beads during the wire arc additive manufacturing process.
    2. Material properties: The physical and chemical properties of the material being used, such as melting point, density, and viscosity, can also impact the formation of beads.
    3. Beam deflection: The deflection of the beam during the wire arc additive manufacturing process can affect the formation of beads.
    4. Arc length: The length of the arc during the wire arc additive manufacturing process can also impact the formation of beads.
    5. Temperature: The temperature of the wire and the workpiece during the wire arc additive manufacturing process can affect the formation of beads.
    6. Wire diameter: The diameter of the wire being used during the wire arc additive manufacturing process can also impact the formation of beads.
    7. Gas pressure: The pressure of the gas used during the wire arc additive manufacturing process can affect the formation of beads.
    8. Feedback control signals: Feedback control signals from sensors placed on the workpiece or the wire can provide information about the formation of beads and can be used as inputs to the machine learning model.

    Now, with those examples, here is the actual chat history and input question.

    Chat History:
    {chat_history}

    Follow Up Input: {question}
    [/INST]
    """
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    #######################################################################
    #ANSWER_PROMPT
    template = """
    [INST]
    Answer the question based only on the following context:
    {context}

    Question: {question}
    [/INST]
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

    #######################################################################
    #DEFAULT_DOCUMENT_PROMPT
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    return CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT


CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT = prompt_templates()

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
  ):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def create_rag(llm, retriever,
              CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT):
    # Instantiate ConversationBufferMemory
    memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
    )

    # First we add a step to load memory
    # This adds a "memory" key to the input object
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )
    # Now we calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm,
    }
    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }
    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }
    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | llm,
        "question": itemgetter("question"),
        "context": final_inputs["context"]
    }
    # And now we put it all together!
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer
    return final_chain, memory

def call_conversational_rag(question, chain, memory):
    # Prepare the input for the RAG model
    inputs = {"question": question}

    # Invoke the RAG model to get an answer
    result = chain.invoke(inputs)

    # Save the current question and its answer to memory for future context
    memory.save_context(inputs, {"answer": result["answer"]})

    # Return the result
    return result

retriever = call_vectordb(embedding_function)

def main(question, llm, retriever):
    CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT = prompt_templates()

    final_chain, memory = create_rag(llm, retriever,
              CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT)

    result= call_conversational_rag(question, final_chain, memory)

    # Convert result to DataFrame
    result_df = pd.DataFrame({
        'question': [question],
        'answer': [result['answer']],
        'contexts': [result['context']]
    })

    print(result)
    return result, result_df


question = input("What is the your question?: ")

main(question, llm, retriever)