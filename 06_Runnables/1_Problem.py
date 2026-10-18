import random

class FakeLLM:
    def __init__(self):
        print('LLM created.')

    def predict(self, prompt):

        response_list = [
            'Dhaka is the capital of Bangladesh',
            'LaLiga is a football league',
            'AI stands for Artificial Intelligence'
        ]

        return {'Response' : random.choice(response_list)}


class FakePromptTemplate:
    def __init__(self, template, input_variable):
        self.template = template
        self.input_variable = input_variable

    def format(self, input_dict):
        return self.template.format(**input_dict)




llm = FakeLLM()
print(llm.predict(prompt='Blah'))

template = FakePromptTemplate(
    template='Write a poem about {topic}',
    input_variable = ['topic']
)

prompt = template.format({'topic':'india'})

print(prompt)



class FakeLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)

        return result['Response']

template = FakePromptTemplate(
    template='Write a song about {topic}',
    input_variable = ['topic']
)

llm = FakeLLM()

chain = FakeLLMChain(llm=llm, prompt=template)

result = chain.run({'topic':'whatever'})
print(result)