import os
import streamlit as st
from gtts import gTTS
import speech_recognition as sr
import tempfile
import base64
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document



# Load environment variables
load_dotenv()


# Streamlit setup
st.set_page_config(page_title="Sandal Voice Chatbot")
st.title("🎙️ Talk to Sandal (Voice Assistant)")
st.markdown("Record your voice and chat with Sandal Agrawal!")



# Step 1: Load and build vector DB from PDF
@st.cache_resource
def get_chroma_db():
    loader = PyPDFLoader("data/About_me.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    db = Chroma.from_documents(
        documents=split_docs,
        embedding=OpenAIEmbeddings(),
        persist_directory="chroma_db"
    )
    return db

# Step 2: Build retriever and memory
db = get_chroma_db()
retriever = db.as_retriever(search_kwargs={"k": 3})
memory = ConversationBufferMemory(return_messages=True)

# Step 3: Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are Sandal Agrawal, an AI-powered version of yourself.
You must answer all questions in the first person as if you are Sandal herself.

Base all your answers on the following context. Be friendly, polite, and confident.
If a question is not covered in the context, you can infer a response that aligns with Sandal's personality and values.

Always speak in a natural, conversational tone as if you are talking to a friend or colleague.

Context:
{context}
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Step 4: Utility for building enhanced context-aware query
def get_chat_history(memory):
    history = memory.load_memory_variables({}).get("chat_history", [])
    messages = []
    for msg in history:
        if msg.type == "human":
            messages.append(f"Human: {msg.content}")
        elif msg.type == "ai":
            messages.append(f"AI: {msg.content}")
    return "\n".join(messages)

def get_context_from_retriever(question, memory):
    chat_history = get_chat_history(memory)
    enhanced_query = f"{chat_history}\nHuman: {question}"
    return retriever.invoke(enhanced_query)

# Step 5: Define final RAG chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
chain = (
    RunnableMap({
        "context": lambda x: get_context_from_retriever(x["question"], memory),
        "question": lambda x: x["question"],
        "chat_history": lambda x: memory.load_memory_variables({}).get("chat_history", []),
    }) | prompt | llm | StrOutputParser()
)

# Step 6: Voice input
audio_bytes = st.audio_input("Click to record your question")

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    # Save temp audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes.read())
        tmp_path = tmp.name

    # Speech-to-text
    recognizer = sr.Recognizer()
    with sr.AudioFile(tmp_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            st.markdown(f"**You said:** {text}")
        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
            os.remove(tmp_path)
            st.stop()
        except sr.RequestError:
            st.error("Speech Recognition service error.")
            os.remove(tmp_path)
            st.stop()

    # RAG + GPT response
    response = chain.invoke({"question": text})
    st.markdown(f"**Sandal says:** {response}")

    # Text-to-speech with gTTS
    tts = gTTS(response)
    tts_fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tts_fp.name)

    # Playback
    st.audio(tts_fp.name, format="audio/mp3")

    # Save chat history
    memory.save_context({"question": text}, {"output": response})

    # Cleanup
    os.remove(tmp_path)
