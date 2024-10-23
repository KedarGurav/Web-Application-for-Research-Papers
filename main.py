import streamlit as st
import arxiv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
###import google.generativeai as geneai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from dotenv import load_dotenv
import os

nltk.download('punkt_tab')


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(layout="wide")

def summarymaker(text):
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)
    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    
    sentences = sent_tokenize(text)
    sentenceValue = dict()
    
    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq
    
    sumValues = 0 
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    average = int(sumValues / len(sentenceValue))

    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence
    return summary

st.title("Comprehensive AI-Powered Web Application for Research Papers ")
st.write("We Demystify Research Papers")

topic = st.text_input("Enter a research topic that you are interested in:")

papers = {}

search = arxiv.Search(
    query=topic,
    max_results=5,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

for result in arxiv.Client().results(search):
    papers[result.title] = [result.entry_id, result.summary, result.pdf_url]

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
output_parser = CommaSeparatedListOutputParser()

template = """
You give a concise and easy to understand summary of a research paper that anyone can understand easily.
Question: Give summary to: {text}
Answer: Keep it short and easy to understand
"""

prompt_template = PromptTemplate(input_variables=["text"], template=template, output_parser=output_parser)
answer_chain = LLMChain(llm=llm, prompt=prompt_template)

for i in papers:
    st.subheader(i)
    st.caption("URL: " + papers[i][2])

    st.write("Feynman Bot's Summary: ", answer_chain.run(papers[i][1]))
    st.write("Original Abstract: ", papers[i][1])
    st.divider()


llm1 = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

output_parser1 = CommaSeparatedListOutputParser()

template1 = """
You answer questions based on a bunch of summaries from research papers we send.
Question: Answer the following question: {text} based on {papers}
Answer: Keep it short and easy to understand
"""

prompt_template1 = PromptTemplate(input_variables=["text", "papers"], template=template1, output_parser=output_parser1)
answer_chain1 = LLMChain(llm=llm1, prompt=prompt_template1)

st.header("Feynman Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    inputs = {
        'text': prompt,
        'papers': ' '.join([summarymaker(papers[i][1]) for i in papers])
    }

    response = answer_chain1.run(inputs)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
