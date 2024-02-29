import subprocess
import json
import pandas as pd
from datasets import Dataset
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from ragas.llms import LangchainLLM
from eval import evaluate
import os
import openai
from ragas.llms import LangchainLLM
from sqlalchemy import column

# Path to the JSON file
file_path = 'scripts/llm_evaluation_set/evaluation.json'

os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_KEY"] = "ab26691b120a4c3ebade1e1d8ba7391f"


azure_model = AzureChatOpenAI(
    deployment_name="faradai-chat",
    model="gpt-35-turbo",
    openai_api_base="https://faradai.openai.azure.com/",
    openai_api_type="azure",
)

# Lists to store questions and ground truths
questions = []
ground_truths = []
answers = []
contexts = []

# Reading the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

    # Iterating through each item in the JSON file
    for item in data:
        questions.append(item['question'])
        ground_truths.append([item['ground_truth']])  # Wrap ground_truth in a list

print("First Question:", questions[0])
print("First Ground Truth:", ground_truths[0])


# Path to the working directory
working_directory = '/Users/hakandemirer/Reengen/reengen-sustain-backend/scripts'

# Iterate through each question
for question in questions:
    # Modify the command for the current question
    command = [
        'python3', 
        'AIAssistant.py', 
        'answer_question', 
        json.dumps({
            "db_settings": {"host": "localhost", "port": 19530},
            "collection_name": "sustain"
        }), 
        json.dumps({
            "question": question,
            "history": [],
            "industry": "",
            "promptType": "QA_PROMPT",
            "userExperience": "EXPERIENCED",
            "modelMode": "HYBRID"
        })
    ]

    # Run the command

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=working_directory)
    stdout, stderr = process.communicate()

    # Decode the output and error
    output = stdout.decode()
    error = stderr.decode()

    # Check for errors
    if process.returncode != 0:
        print("Error running script for question:", question)
        print("Error details:", error)
    else:
        # Process the output
        output_dict = json.loads(output)
        answers.append(output_dict['result']['answer'])
        page_contents = [doc['page_content'] for doc in output_dict['result']['source_documents']]
        contexts.append(page_contents)

print("First Answer:", answers[0])
print("First Context:", contexts[0])


data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

from ragas.metrics import (
    faithfulness, 
    # answer_relevancy, 
    # context_recall, 
    context_precision, 
    answer_similarity, 
    # answer_correctness
)


metrics=[
        faithfulness, # 4-8 sn --- 1:08 dk 
        # answer_relevancy, # 11-34 sn 
        context_precision, # 3 sn --- 41 sn
        #context_recall, # 5 sn
        answer_similarity, # 12 sn  --- 1:01 dk 
    ]

# wrapper around azure_model-
ragas_azure_model = LangchainLLM(azure_model)


# patch the new RagasLLM instance
# answer_relevancy.llm = ragas_azure_model
answer_similarity.llm = ragas_azure_model

# init and change the embeddings
# only for answer_relevancy
azure_embeddings = AzureOpenAIEmbeddings(
    deployment="faradai-chat-ada",
    model="gpt-35-turbo",
    openai_api_base="https://faradai.openai.azure.com/",
    openai_api_type="azure",
)
# embeddings can be used as it is
# answer_relevancy.embeddings = azure_embeddings
answer_similarity.embeddings = azure_embeddings

for m in metrics:
    m.__setattr__("llm", ragas_azure_model)

##### parallel evaluation #####
datasets = []
for i in range(len(questions)):
    dataset = Dataset.from_dict({
        "question": [questions[i]],
        "ground_truths": [ground_truths[i]],
        "answer": [answers[i]],
        "contexts": [contexts[i]]
    })
    datasets.append(dataset)

from concurrent.futures import ThreadPoolExecutor
def evaluate_row(one_set):
    # Evaluate metrics for the single row
    result = evaluate(dataset=one_set, metrics=metrics)
    return result.to_pandas().iloc[0]

with ThreadPoolExecutor() as executor:
    # Evaluate each row in parallel
    futures = [executor.submit(evaluate_row, one_set) for one_set in datasets]

    # Collect results as they complete
    results = [future.result() for future in futures]

combined_df = pd.DataFrame(results)
overall_score = round(combined_df.iloc[:, 4:].mean().sum() / len(metrics), 3)
combined_df.to_excel("scripts/evals/output - {}.xlsx".format(overall_score), index=False) 
###############################################


# ##### serial-normal evaluation #####    
# result = evaluate(dataset=dataset, metrics=metrics)
# df =result.to_pandas()
# ##
# num_metrics=len(metrics)
# overall_score=round(df.iloc[:,4:].mean().sum()/num_metrics, 3)
# ##
# ## Save the DataFrame to an Excel file
# df.to_excel("scripts/evals/output - {}.xlsx".format(overall_score), index=False) 
# ######
