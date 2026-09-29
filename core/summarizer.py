import os
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_llm():
    return ChatGroq(
        model="qwen/qwen3.8-27b",  
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    # Respect Mistral's 1 req/sec rate limit by sleeping between chunks
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            time.sleep(2)  # Pause to avoid 429 Rate Limit Exceeded
        summary_piece = map_chain.invoke({"text": chunk})
        chunk_summaries.append(summary_piece)

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points.",
        ),
        ("human", "{text}"),
    ])
    combined_chain = combined_prompt | llm | StrOutputParser()

    time.sleep(2)  # Pause before the final reduction call
    return combined_chain.invoke({"text": combined})


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Based on the meeting transcript, generate a short professional meeting title "
            "(max 8 words). Only return the title, nothing else.",
        ),
        ("human", "{text}"),
    ])
    title_chain = title_prompt | llm | StrOutputParser()

    return title_chain.invoke({"text": transcript[:2000]})