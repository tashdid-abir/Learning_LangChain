import random
from abc import ABC, abstractmethod


class Runnable(ABC):

    @abstractmethod
    def invoke(input_data):
        pass



class FakeLLM(Runnable):
    def __init__(self):
        print('LLM created.')

    def invoke(self, prompt):
        response_list = [
            'Dhaka is the capital of Bangladesh',
            'LaLiga is a football league',
            'AI stands for Artificial Intelligence'
        ]

        return {'response': random.choice(response_list)}

    def predict(self, prompt):

        print('This method is going away. Use .invoke()')

        response_list = [
            'Dhaka is the capital of Bangladesh',
            'LaLiga is a football league',
            'AI stands for Artificial Intelligence'
        ]

        return {'response': random.choice(response_list)}
    


class FakePromptTemplate(Runnable):
    def __init__(self, template, input_variable):
        self.template = template
        self.input_variable = input_variable

    def invoke(self, input_dict):
        return self.template.format(**input_dict)
    
    def format(self, input_dict):

        print('This method is going away. Use .invoke()')
        return self.template.format(**input_dict)


class FakeOutputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data):
        return input_data['response']



class RunnableConnector(Runnable):

    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):

        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

        return input_data



template = FakePromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variable=['length','topic']
)

llm = FakeLLM()

parser = FakeOutputParser()


chain = RunnableConnector([template, llm, parser])

result = chain.invoke({'length':'long', 'topic':'Bangladesh'})

print(result)



template1 = FakePromptTemplate(
    template='Write a joke about a {topic}',
    input_variable=['topic']
)

template2 = FakePromptTemplate(
    template='Write a summary about the following joke {response}',
    input_variable=['response']
)

llm = FakeLLM()

parser = FakeOutputParser()

chain1 = RunnableConnector([template1, llm])

chain2 = RunnableConnector([template2, llm, parser])

final_chain = RunnableConnector([chain1, chain2])

print(final_chain.invoke({'topic': 'blah'}))