# YouTube RAG

A Retrieval-Augmented Generation (RAG) system for answering questions from YouTube video transcripts using embeddings and LLMs.  
This project includes a **backend API** built with FastAPI and can be extended with a **browser extension** or other clients.

---

## 🚀 Features
- Extracts YouTube video transcripts automatically.
- Generates embeddings for semantic search.
- Uses LLMs to provide context-aware answers.
- FastAPI backend ready for deployment (Render/any free hosting).
- Modular structure for future improvements.

---

## 📂 Project Structure
```
youtubeRag/
│── backend/
│   ├── main.py              # FastAPI entry point
│   ├── requirements.txt     # Python dependencies
│   ├── utils/               # Helper functions
│── extension/               # Browser extension 
│── README.md                # Project documentation
```

---

## ⚙️ Backend Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Nitinpratap22061/youtubeRag.git
cd youtubeRag/backend
```

### 2️⃣ Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate    # On Windows

pip install -r requirements.txt
```

### 3️⃣ Run FastAPI Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Now your backend runs at: `http://127.0.0.1:8000`

---

## 🌐 Deployment (Render)

1. Push your code to GitHub.
2. Go to [Render](https://render.com/), create a **new web service**.
3. Connect your repo and select the `backend` folder.
4. Add **Build Command**:
   ```bash
   pip install -r requirements.txt
   ```
5. Add **Start Command**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
6. Deploy 🎉

---

## 🔑 Environment Variables

Create a `.env` file inside `backend/`:
```
GROQ_API_KEY=
EMBED_MODEL=
GROQ_MODEL=
CHUNK_SIZE=
CHUNK_OVERLAP=
TOP_K=
MAX_CACHE=

```

Make sure to add the key in Render **Environment Settings** too.



## 📦 Requirements
- Python 3.10+
- FastAPI
- Uvicorn
- LangChain / SentenceTransformers
- OpenRouter API key

---

## 🤝 Contribution
Feel free to fork this repo and create pull requests.

---

## 📜 License
MIT License © 2025 [Nitin Pratap](https://github.com/Nitinpratap22061)
