"""End-to-end RAG test on the remote host: fresh collection -> ingest -> hybrid retrieve."""
import asyncio

from qdrant_client import AsyncQdrantClient
from researchops.rag.ingest import ingest_pdf
from researchops.rag.retriever import Retriever


async def main() -> None:
    # Fresh collection: drop any previously-ingested (wrong) paper data.
    client = AsyncQdrantClient(url="http://127.0.0.1:6333", check_compatibility=False)
    if await client.collection_exists("papers"):
        await client.delete_collection("papers")
        print("dropped existing 'papers' collection")
    await client.close()

    print("=== ingest ===")
    n = await ingest_pdf("/root/autodl-tmp/restormer.pdf")
    print("ingested chunks:", n)

    print("=== search ===")
    retriever = Retriever()
    try:
        queries = [
            "What is Restormer and how does it improve image restoration?",
            "What PSNR does Restormer achieve on CBSD68 for Gaussian color denoising?",
            "How does Restormer's MDTA multi-head attention achieve linear complexity?",
        ]
        for query in queries:
            print(f"\nQ: {query}")
            results = await retriever.retrieve(query, top_k=5)
            for i, r in enumerate(results, 1):
                c = r.chunk
                print(f"  [{i}] score={r.score:.4f}  p{c.page} ({c.section})")
                print(f"      {c.text[:140].strip()}")
    finally:
        await retriever.close()

    print("\nE2E_OK")


if __name__ == "__main__":
    asyncio.run(main())
