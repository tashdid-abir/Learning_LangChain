import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt

from dotenv import load_dotenv

import streamlit as st

load_dotenv()
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ["OPENROUTER_MODEL"]

model = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)


st.header('Research Tool')

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
        "Denoising Diffusion Probabilistic Models"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Mathematical",
        "Code-Oriented",
        "Storytelling"
    ]
)

lenght_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)

template = load_prompt(r'H:\LangChain\Prompts\template.json')


if st.button('Summarize'):

    chain = template | model

    result = chain.invoke({
    'paper_input' : paper_input,
    'style_input' : style_input,
    'lenght_input': lenght_input
    })

    st.write(result.content)