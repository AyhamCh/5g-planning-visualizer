"""
rag_pipeline.py
===============
Pipeline RAG 5G — Script standalone (une seule exécution)

Étapes :
  1. Extraction texte des PDFs (PyMuPDF)
  2. Chunking des textes (LangChain RecursiveCharacterTextSplitter)
  3. Indexation ChromaDB (SentenceTransformer embeddings)
  4. Test retrieval + Q&A via Ollama (qwen3:8b)

Usage :
  python rag_pipeline.py                    # build + test interactif
  python rag_pipeline.py --build-only       # build sans Q&A
  python rag_pipeline.py --query-only       # Q&A sur DB existante
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz                                          # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from ollama import chat

# =============================================================================
# CONFIGURATION — adaptez ces chemins selon votre environnement
# =============================================================================

PDF_DIR     = Path.home() / "Downloads" / "knowledge_base"
TXT_DIR     = Path.home() / "Downloads" / "knowledge_base_txt"
PROJECT_SUMMARY = Path.cwd() / "outputs" / "agents_summary.txt"
CHROMA_PATH = "./chroma_db"
COLLECTION  = "knowledge_base"
EMBED_MODEL = "intfloat/multilingual-e5-base"
LLM_MODEL   = "qwen3:8b"

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
N_RESULTS     = 5      # nombre de chunks retournés par le retrieval


# =============================================================================
# ÉTAPE 1 — EXTRACTION PDF → TXT
# =============================================================================

def extract_pdfs() -> int:
    """Extrait le texte de tous les PDFs dans PDF_DIR vers TXT_DIR."""
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    files = list(PDF_DIR.rglob("*.pdf"))

    if not files:
        print(f"[!] Aucun PDF trouvé dans : {PDF_DIR}")
        return 0

    print(f"\n[1/3] EXTRACTION PDF ({len(files)} fichier(s))")
    print("-" * 50)

    total_chars = 0

    for pdf_file in files:
        print(f"  Lecture : {pdf_file.name}")
        text = ""

        doc = fitz.open(pdf_file)
        for page in doc:
            text += page.get_text()
        doc.close()

        output_file = TXT_DIR / f"{pdf_file.stem}.txt"
        output_file.write_text(text, encoding="utf-8")

        total_chars += len(text)
        print(f"    → {len(text):,} caractères → {output_file.name}")

    print(f"  Total : {total_chars:,} caractères extraits")
    return len(files)



def extract_project_summary() -> int:
    """
    Ajoute le résumé des agents du projet
    dans le dossier TXT utilisé par le RAG
    """

    if not PROJECT_SUMMARY.exists():
        print("[!] agents_summary.txt introuvable")
        return 0

    TXT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = TXT_DIR / PROJECT_SUMMARY.name

    text = PROJECT_SUMMARY.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    output_file.write_text(
        text,
        encoding="utf-8"
    )

    print(
        f"  ✓ Document projet ajouté : {output_file.name}"
    )

    return 1

# =============================================================================
# ÉTAPE 2 + 3 — CHUNKING + INDEXATION CHROMADB
# =============================================================================

def build_vectordb() -> int:
    """Chunk les .txt et indexe dans ChromaDB. Retourne le nombre de chunks."""
    txt_files = list(TXT_DIR.glob("*.txt"))

    if not txt_files:
        print(f"[!] Aucun .txt trouvé dans : {TXT_DIR}")
        print("    Lancez d'abord l'extraction PDF (sans --query-only).")
        return 0

    print(f"\n[2/3] CHUNKING + INDEXATION ({len(txt_files)} fichier(s))")
    print("-" * 50)

    # Aperçu du découpage
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    total_chunks = 0
    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        chunks = splitter.split_text(text)
        print(f"  {txt_file.name} → {len(chunks)} chunks")
        total_chunks += len(chunks)

    print(f"  Total : {total_chunks} chunks à indexer")

    # Modèle embeddings
    print(f"\n  Chargement modèle embeddings : {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    # ChromaDB
    print(f"  Connexion ChromaDB : {CHROMA_PATH}")
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION)

    # Indexation
    doc_count = 0
    for txt_file in txt_files:
        text   = txt_file.read_text(encoding="utf-8", errors="ignore")
        chunks = splitter.split_text(text)

        embeddings = model.encode(chunks).tolist()
        ids        = [f"{txt_file.stem}_{i}" for i in range(len(chunks))]
        metadatas  = [{"source": txt_file.name} for _ in chunks]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        doc_count += len(chunks)
        print(f"  ✓ {txt_file.name} ({len(chunks)} chunks indexés)")

    print(f"\n  Base vectorielle prête — {doc_count} chunks dans '{COLLECTION}'")
    return doc_count


# =============================================================================
# ÉTAPE 4 — TEST RETRIEVAL + Q&A INTERACTIF
# =============================================================================

def run_qa_loop() -> None:
    """Boucle Q&A interactive : retrieval ChromaDB + génération Ollama."""
    print(f"\n[3/3] Q&A INTERACTIF (modèle : {LLM_MODEL})")
    print("-" * 50)

    print(f"  Chargement modèle embeddings : {EMBED_MODEL}")
    model  = SentenceTransformer(EMBED_MODEL)

    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION)

    print("  Base vectorielle connectée.")
    print("  Tapez 'exit' pour quitter.\n")

    while True:
        question = input("Question : ").strip()

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Au revoir.")
            break

        # Retrieval
        query_embedding = model.encode(question).tolist()
        results         = collection.query(
            query_embeddings=[query_embedding],
            n_results=N_RESULTS,
        )
        context = "\n\n".join(results["documents"][0])

        # Affichage sources
        sources = list({m["source"] for m in results["metadatas"][0]})
        print(f"\n  Sources : {', '.join(sources)}")

        # Génération LLM
        # APRÈS
        prompt = f"""You are a 5G telecommunications expert.
        Use the provided context as your primary source. You may also use your general knowledge 
        to complete or structure the answer, but always prioritize the context.

        Context:
        {context}

        Question:
        {question}

        Answer:"""

        response = chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"think": False},
        )

        print(f"\nRéponse :\n{response['message']['content']}\n")
        print("-" * 50)


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline RAG 5G — extraction PDF → ChromaDB → Q&A",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Extraction + indexation uniquement (pas de Q&A)",
    )
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Q&A uniquement (suppose DB déjà construite)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PIPELINE RAG 5G")
    print("=" * 60)

    if not args.query_only:

        n_pdfs = extract_pdfs()

        n_project = extract_project_summary()

        if n_pdfs == 0 and n_project == 0:
            sys.exit(1)

        n_chunks = build_vectordb()
        if n_chunks == 0:
            sys.exit(1)

    if not args.build_only:
        run_qa_loop()


if __name__ == "__main__":
    main()