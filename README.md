# Banking RAG — Human Review System

## Overview

A banking-focused Retrieval-Augmented Generation system with
hybrid retrieval, reranking, confidence analysis, policy conflict
detection, human review, evidence tracking, and auditability.

## Architecture

User Question
↓
Product Detection
↓
Query Rewriting
↓
Hybrid Retrieval
├── Vector Retrieval
└── BM25 Retrieval
↓
Reciprocal Rank Fusion
↓
Reranker
↓
Confidence & Conflict Detection
↓
Quality Gate
↓
Decision Layer
├── PROCEED
└── MANUAL_REVIEW
↓
Grounded Answer
↓
Evidence & Citations
↓
Audit Trail

## Human Review Workflow

MANUAL_REVIEW
↓
Review Queue
↓
Reviewer
↓
APPROVED / REJECTED
↓
Reviewer Comment
↓
Resolution Timestamp
↓
Audit Trail

## Backend

- FastAPI
- Pydantic
- Qdrant
- Hybrid Retrieval
- BM25
- Reranking
- Ollama

## Frontend

- React
- Vite
- Human Review Dashboard

## Main Features

- Banking policy retrieval
- Product-aware retrieval
- Query rewriting
- Hybrid search
- Reciprocal Rank Fusion
- Cross-encoder reranking
- Confidence analysis
- Policy conflict detection
- Manual review queue
- Human approval/rejection
- Evidence display
- Reviewer comments
- Resolution timestamps
- Dashboard metrics
- Search and status filtering
- API validation
- Audit logging

## Run Backend

```powershell
cd "C:\Users\amith\Desktop\Confidential\Misc Projects\P10\rag_projects"

python -m uvicorn api:app --reload