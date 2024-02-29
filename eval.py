from AMEngineer import *
import pandas as pd
import ragas
import json
from datasets import Dataset
from ragas import evaluate
import asyncio
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

question = input("What is the your question?: ")
result, result_df = main(question, llm, retriever)

def full_main(result, result_df):
    ground_truth = """
    As of my last knowledge update in January 2022, there isn't a widely recognized and specific process or acronym known as "DED" in a universal context. However, it's possible that the term or acronym has been introduced or gained significance after that date, or it could be specific to a particular field, industry, or context.

    To provide a more accurate explanation, could you please provide additional details or specify the context in which you encountered the term "DED"? This information will help me provide a more relevant and detailed response based on the specific meaning or process associated with "DED" in your inquiry.
    """

    # Inference
    answers = result_df['answer'].values.tolist()
    contexts = [docs.page_content for docs in retriever.get_relevant_documents(question)]

    # To dict
    data = {
        "question": [question],
        "answer": answers,
        "contexts": [contexts],
        "ground_truth": [ground_truth]  # Add 'ground_truth' column
    }

    # Convert dict to dataset
    dataset = Dataset.from_dict(data)

    # Evaluation ve diğer işlemler buraya taşınabilir
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    df = result.to_pandas()
    print(df.head())


full_main(result, result_df)