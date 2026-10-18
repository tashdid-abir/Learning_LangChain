from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are an cricket expert.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

with open('Prompts/6_chat_history.txt') as f:
    chat_history.extend(f.readlines())

prompt = chat_template.invoke({
    'chat_history' : chat_history,
    'query' : 'What is a wicket ?'
})

print(prompt)