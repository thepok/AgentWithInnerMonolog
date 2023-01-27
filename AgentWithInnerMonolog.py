import string
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.cache import SQLiteCache
import langchain

langchain.llm_cache=SQLiteCache(".langchain.db")

#generate a class containing a method for each prompt
class AgentWithInnerMonolog():
    def __init__(self) -> None:
        self.ListThoughtsOfInnerMonologPrompt = PromptTemplate(
        input_variables=["log"],
        template="""Read the Log of two experts between >>> and <<< and list the thoughts below.
>>>
        {log}
<<<
 Now list all the thougths of the experts in two or three sentences per thought in the text above in great detail. Try to Gruop them under subtopics:
-""")
        self.llm = OpenAI(temperature=0.2, max_tokens=1000)
        self.ListThoughtsChain = LLMChain(llm=self.llm, prompt=self.ListThoughtsOfInnerMonologPrompt)

        self.CoherentTextFromThoughtsPrompt = PromptTemplate(
        input_variables=["thoughts"],
        template="""Read the Log of thoughts of an expert between >>> and <<< and write a coherent text from the thoughts below.

>>>
{thoughts}
<<<

Now write a coherent text from the thoughts above:
""")
        self.CoherentTextFromThoughtsChain = LLMChain(llm=self.llm, prompt=self.CoherentTextFromThoughtsPrompt)


    def GetTextOnTopic(self, topic, searchdepth=3):
        innerMonolog=InnerMonolog(cache=False)

        inner_monolog_text=innerMonolog.MakeInnerMonolog(topic = topic, rounds=searchdepth)
        print("Inner Monolog:")
        print(inner_monolog_text)
        thoughts = self.ListThoughtsChain.predict(log=inner_monolog_text)
        print("Thougths of Monolog:")
        print(thoughts)
        result = self.CoherentTextFromThoughtsChain.predict(thoughts=thoughts)
        return result

class InnerThought():
    def __init__(self, role, tought):
        self.role=role
        self.tought=tought
    
    def __str__(self):
        return self.role + ": " + self.tought

class InnerMonolog():
    def __init__(self, cache=True):
        self.llm=None
        if(cache):
            self.llm=OpenAI(temperature=0.7, max_tokens=1000)
        else:
            self.llm=OpenAI(temperature=0.7, max_tokens=1000, cache=False)
            
        #current list of inner thoughts
        self.currentThoughts=[]

        NextQuestionPrompt = PromptTemplate(
        input_variables=["topic", "log"],
        template="""Two Experts talk about the following Topic:
{topic}

The Experts have read the Text above, and now talk about there own ideas, that they want to add to the text.
Current Conversation:
{log}

Generate the next three Questions for the Experts to talk about. Demand a detailed answer from the Expert.
ExpertQuestioner:"""
        )
        self.questionChain = LLMChain(llm=self.llm, prompt=NextQuestionPrompt)

        NextAnswerPrompt = PromptTemplate(
        input_variables=["topic", "log"],
        template="""
Two Experts talk about the following Topic:
{topic}

The Experts have read the Text above, and now talk about there own ideas, that they want to add to the text.
The Experts talk back and forth.
log:{log}
Expert Answerer:""")
        self.answerChain = LLMChain(llm=self.llm, prompt=NextAnswerPrompt)

    def ThoughtsToLog(self):
        #return the current list of inner thoughts as one string by calling the tostring method of 
        #each inner thought and joining them with a new line
        
        return '\n'.join(thought.__str__() for thought in self.currentThoughts)

    def MakeInnerMonolog(self, topic, rounds=3):
        #rounds times do:
        #   ask the questioner for a question
        #   ask the answerer for an answer
        #   add the question and answer to the current list of inner thoughts
        #return the current list of inner thoughts as one string by calling the ThoughtsToLog method
        
        
        for i in range(rounds):
            questions=self.questionChain.predict(topic=topic, log=self.ThoughtsToLog())
            self.currentThoughts.append(InnerThought("Questioner", questions))
            answers=self.answerChain.predict(topic=topic, log=self.ThoughtsToLog())
            self.currentThoughts.append(InnerThought("Answerer", answers))
        return self.ThoughtsToLog()

if __name__ == "__main__":
    #topic="The invention of AGI and the stockmarket"
    #topic="Star Trek"
    #topic = "personal finances of an immortal person"
    #topic = "germany in the 22nd century" TODO it forgets that its about germany
    topic = "zombie company"
    agent=AgentWithInnerMonolog()
    answer = agent.GetTextOnTopic(topic, searchdepth=5)
    print(answer)
