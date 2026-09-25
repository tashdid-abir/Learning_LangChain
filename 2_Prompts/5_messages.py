from langchain_core.prompts import ChatPromptTemplate

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# chat_template = ChatPromptTemplate([
#     SystemMessage(content='You are a helpful {domain} expert.'),
#     HumanMessage(content="Explain in simple terms what {topic} is.")
# ])


chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain in simple terms what {topic} is."),
])


prompt = chat_template.invoke({
    'domain' : 'cricket',
    'topic'  : 'Out'
})

print(prompt)