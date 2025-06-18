import streamlit as st
from gtts import gTTS
import speech_recognition as sr
import tempfile
import os
import base64
from langchain_core.runnables import RunnableMap
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Setup LLM and DB
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
db = Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings())
retriever = db.as_retriever(search_kwargs={"k": 3})
memory = ConversationBufferMemory(return_messages=True)

# Prompt
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

# History retriever
def get_chat_history(memory):
    history = memory.load_memory_variables({}).get("chat_history", [])
    messages = []
    for msg in history:
        if msg.type == "human":
            messages.append(f"Human: {msg.content}")
        elif msg.type == "ai":
            messages.append(f"AI: {msg.content}")
    return "\n".join(messages)

# Get context
def get_context_from_retriever(question, memory):
    chat_history = get_chat_history(memory)
    enhanced_query = f"{chat_history}\nHuman: {question}"
    return retriever.invoke(enhanced_query)

# Final chain
chain = (
    RunnableMap({
        "context": lambda x: get_context_from_retriever(x["question"], memory),
        "question": lambda x: x["question"],
        "chat_history": lambda x: memory.load_memory_variables({}).get("chat_history", []),
    }) | prompt | llm | StrOutputParser()
)

# Streamlit app UI
st.set_page_config(page_title="Sandal Voice Chatbot")
st.title("🎙️ Talk to Sandal (Voice Assistant)")
st.markdown("Record your voice and chat with Sandal Agrawal!")

# Record voice
audio_bytes = st.audio_input("Click to record your question")

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    # Save temp audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes.read())
        tmp_path = tmp.name

    # Speech recognition
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

    # Convert to speech (gTTS)
    tts = gTTS(response)
    tts_fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tts_fp.name)

    # Play back
    st.audio(tts_fp.name, format="audio/mp3")

    # Save to memory
    memory.save_context({"question": text}, {"output": response})

    # Cleanup
    os.remove(tmp_path)
