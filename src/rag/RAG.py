import dspy
import logging
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from pymilvus import MilvusClient, DataType

from utils.rag import Chunk
from utils.rag import chunk_paper
from utils.embedding import embed
from utils.log import init_logging
from utils.paper_schema import Paper
from utils.lm import openrouter_config
from config import EMBED_MODEL, ANSWER_MODEL, EMBED_DIMENSION, RETRIEVAL_LIMIT

openrouter_config(model_str=ANSWER_MODEL)
logger = logging.getLogger(__name__)

class _AnswerSignature(dspy.Signature):
    """
    You are a precise question-answering assistant. Answer the query using ONLY the retrieved context provided.

    Rules:
    - If the context contains the answer, set has_answer to True and answer concisely and accurately.
    - If the context does not contain enough information, set has_answer to False.
    - Do not use any prior knowledge outside of the retrieved context.
    - Do not speculate or infer beyond what is explicitly stated in the context.
    - If the answer spans multiple chunks, synthesize them into a single coherent response.
    """

    query: str = dspy.InputField(desc="The question being asked.")
    retrieved_context: str = dspy.InputField(desc="Retrieved context from an external knowledge base.")

    has_answer: bool = dspy.OutputField(desc="True if the context contains enough information to answer the query, False otherwise.")
    response: str = dspy.OutputField(desc="Your answer to the query given the retrieved context. Only populated if has_answer is True.")

_answer = dspy.Predict(_AnswerSignature)

class RAG:
    '''
    a basic vectorRAG implementation
    '''

    def __init__(self, embed_model: str = EMBED_MODEL):
        self.retrieval_limit = RETRIEVAL_LIMIT
        self.embed_model = embed_model
        self.milvus_client = MilvusClient(uri="http://localhost:19530")
        self.collection_name = "paper_chunks"
        if not self.milvus_client.has_collection(self.collection_name):
            self._create_collection()

    def _create_collection(self): 
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=36, is_primary=True, auto_id=False)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=EMBED_DIMENSION)
        schema.add_field(field_name="section_header", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=12000)
        schema.add_field(field_name="source_doc", datatype=DataType.VARCHAR, max_length=32)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256}
        )

        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    async def ingest(self, paper: Paper) -> None:
        '''
        ingest a preprocessed paper document into milvus db
        '''
        logger.info(f"Starting ingestion from {paper.front.hash}...")

        # chunk paper
        chunks = chunk_paper(paper)

        # embed chunks
        async def task(sem: asyncio.Semaphore, chunk) -> None:
            async with sem:
                chunk.embedding = await embed(chunk.text)
        sem = asyncio.Semaphore(5)
        tasks = [task(sem, chunk) for chunk in chunks]
        await tqdm_asyncio.gather(*tasks)

        # add chunks to db
        data = [chunk.model_dump() for chunk in chunks]
        res =self.milvus_client.upsert(
            collection_name=self.collection_name,
            data=data
        )

        logger.info(f"Upserted {res['upsert_count']} chunks to '{self.collection_name}'")
    
    async def retrieve(self, query: str) -> str:
        '''
        retrieves relevant context chunks
        '''
        logger.info("Embedding query...")
        query_embed = await embed(query)
        logger.info("Searching Milvus db...")
        search_res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_embed],
            limit=self.retrieval_limit,
            search_params={"metric_type": "COSINE"},
            output_fields=["text", "source_doc", "section_header"],
        )
        
        chunks = []
        for hits in search_res:
            for hit in hits:
                chunk = Chunk.model_validate(hit.entity)
                chunks.append(chunk)
        
        return "\n".join(chunks)
    
    async def query(self, query: str) -> str | None:
        '''
        augments the query using retrieved context from the knowledge bas
        and generates a response
        '''
        context = await self.retrieve(query)
        if context:
            result = await _answer(query=query, retrieved_context=context)
            if result.has_answer:
                return result.response
            else:
                raise Exception(f"No answer for {query} using context:\n\n{context}")
        else:
            raise Exception(f"No context retrieved for {query}")
            
    def reset(self) -> None:
        '''
        clears the internal knowledge base
        '''
        self.milvus_client.drop_collection(self.collection_name)
        self._create_collection()
        logger.info(f"Milvus collection {self.collection_name} cleared")
    
async def test(test_file: Path) -> None:
    with open(test_file, "r") as f:
        paper = Paper.model_validate_json(f.read())
    rag = RAG()
    rag.reset()
    await rag.ingest(paper)
    context_chunks = await rag.retrieve("What is VanA?")
    print(context_chunks)

if __name__ == "__main__":
    init_logging()
    test_file = Path("/Users/karenxu/Documents/Code/USRA/datasets/papers_test_json/0c7e0602-7c45-1014-93c3-be5d83700d97/paper/720744.json")
    asyncio.run(test(test_file))