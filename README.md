# AI Video Assistant & Meeting Intelligence

An automated pipeline designed to transcribe, structure, and query discussions from video recordings, meeting streams, and audio files. The system handles media extraction, executes multi-stage chunked summarization via high-throughput LPUs, and provisions an embedded vector store for Retrieval-Augmented Generation (RAG) over meeting transcripts.

---

## Overview

Processing lengthy recordings manually to capture action items and technical decisions is time-intensive. This project automates the entire lifecycle:
1. Ingests raw YouTube URLs or local media containers (`.mp4`, `.webm`, `.wav`).
2. Normalizes and chunks the audio streams to optimize inference limits.
3. Transcribes multilingual and mixed-dialect speech (English and Hinglish).
4. Employs recursive map-reduce prompting to distill conversations without context overflow.
5. Indexes transcript embeddings into ChromaDB for interactive semantic search and grounded QA.

---

## Architecture Pipeline

```text
[ Input Media: YouTube URL / Local Audio ]
                   │
                   ▼
       Audio Ingestion & Extraction (yt-dlp / FFmpeg)
                   │
                   ▼
     Normalized Audio Chunks (16 kHz Mono WAV)
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
Local Whisper Engine    Sarvam AI API
  (English Audio)     (Hinglish / Indic)
        └──────────┬──────────┘
                   ▼
            Full Transcript
                   │
         ┌─────────┴────────────────────────┐
         ▼                                  ▼
 Map-Reduce Synthesis               Vector Store Indexing
 (Groq / Qwen 27B LPU)              (HuggingFace Embeddings)
         │                                  │
         ├─ Executive Summary               ▼
         ├─ Action Items & Owners       ChromaDB
         ├─ Key Decisions                   │
         └─ Open Inquiries                  ▼
                                     RAG Question Answering
                                     (Streamlit Web Interface)
