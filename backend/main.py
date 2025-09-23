import os, time
from typing import Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from pinecone import Pinecone as PineconeClient


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing. Set it in .env")

EMBED_MODEL = os.getenv("EMBED_MODEL", "embed-english-v3.0") 
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_CACHE = int(os.getenv("MAX_CACHE", "16"))


embeddings = CohereEmbeddings(model=EMBED_MODEL, cohere_api_key=COHERE_API_KEY)
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.2)

prompt = PromptTemplate(
    template="""You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, reply "don't know".

{context}
Question: {question}
""",
    input_variables=["context", "question"],
)

# Init Pinecone client
pc = PineconeClient(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# In-memory cache of Pinecone vectorstore per video
_vector_cache: Dict[str, Dict[str, Any]] = {}


def _trim_cache():
    if len(_vector_cache) <= MAX_CACHE:
        return
    oldest = sorted(_vector_cache.items(), key=lambda kv: kv[1]["at"])[0][0]
    _vector_cache.pop(oldest, None)




def fetch_transcript_text(video_id: str) -> str:
    try:
        ytt_api = YouTubeTranscriptApi()
        snippets = ytt_api.fetch(video_id)

        parts = []
        for sn in snippets:
            if isinstance(sn, dict):
                parts.append(sn.get("text", ""))
            else:
                txt = getattr(sn, "text", None)
                if txt is None:
                    try:
                        parts.append(sn.get("text", ""))
                    except Exception:
                        parts.append(str(sn))
                else:
                    parts.append(txt)
        text = " ".join(p for p in parts if p)
        return text.strip()
    except TranscriptsDisabled:
        raise HTTPException(status_code=400, detail="Transcript is disabled for this video.")
    except NoTranscriptFound:
        raise HTTPException(status_code=404, detail="No transcript found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcript fetch error: {e}")



def get_or_build_vectorstore(video_id: str):
    now = time.time()
    if video_id in _vector_cache:
        _vector_cache[video_id]["at"] = now
        return _vector_cache[video_id]["vs"]

    transcript = fetch_transcript_text(video_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Empty transcript.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(transcript)
    docs = [Document(page_content=c) for c in chunks]

    vs = Pinecone.from_documents(docs, embeddings, index_name=PINECONE_INDEX)
    _vector_cache[video_id] = {"vs": vs, "at": now, "n_chunks": len(chunks)}
    _trim_cache()
    return vs



def build_chain(vs):
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

    def _format(docs):
        return "\n\n".join(d.page_content for d in docs)

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(_format),
        "question": RunnablePassthrough(),
    })
    parser = StrOutputParser()
    return parallel | prompt | llm | parser



app = FastAPI(title="YouTube RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    video_id: str
    question: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask(payload: AskRequest):
    vs = get_or_build_vectorstore(payload.video_id)
    chain = build_chain(vs)
    answer = chain.invoke(payload.question)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn, os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
