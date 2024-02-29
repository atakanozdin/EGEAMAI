import os
from langchain_openai import OpenAI
os.environ["OPENAI_API_KEY"] = "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"

def PainterRun():
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
    from langchain_openai import OpenAI

    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["image_desc"],
        template="Generate a detailed prompt to generate an image based on the following description: {image_desc}",
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    image_url = DallEAPIWrapper().run(chain.run("halloween night at a haunted museum"))

    print(image_url)


PainterRun()