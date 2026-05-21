import json
import dspy
import logging
import numpy as np
from pathlib import Path
from config import EMBED_MODEL, ANSWER_MODEL
from utils.models import Paper
from utils.embed import embed_with_retry

logger = logging.getLogger(__name__)

ANSWER_LM = dspy.LM(ANSWER_MODEL)

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

answer = dspy.Predict(_AnswerSignature)

class RAG:
    '''
    a basic vectorRAG implementation
    '''

    def __init__(self, quantile: float = 0.95, embed_model: str = EMBED_MODEL):
        self.quantile = quantile
        self.embed_model = embed_model
        self.kb_text = []
        self.kb_embeddings = []
    
    def ingest(self, preprocessed_file: Path) -> None:
        '''
        ingest a preprocessed paper document into the knowledge base
        '''
        logger.info(f"Starting ingestion from {preprocessed_file.name}...")
        with open(preprocessed_file, "r") as f:
            data = json.load(f)
        paper = Paper(**data)
        for chunk in paper.abstract + paper.body:
            self.kb_text.append(chunk.text)
            self.kb_embeddings.append(chunk.embeddings)
        
        assert len(self.kb_text) == len(self.kb_embeddings)
        logger.info(f"Done ingestion from {preprocessed_file}")
    

    async def query(self, query: str, context: str) -> str | None:
        '''
        augments the query using retrieved context from the knowledge base
        and generates a response
        '''
        context = self._retrieve(query)
        if context:
            result = answer(query=query, retrieved_context=context)
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
        self.kb_text = []
        self.kb_embeddings = []
        logger.info(f"VectorRAG knowledge base cleared")

    async def _retrieve(self, query: str) -> str:
        query_embed = await embed_with_retry(input_=[query], output_dim=len(self.kb_embeddings[0]))

        sim_matrix = self._cosine_sim(query_embed)
        sim_thresh = np.quantile(sim_matrix, self.quantile)
        sim_indices = np.where(sim_matrix >= sim_thresh)[0]
        sim_indices = sim_indices[np.argsort(sim_matrix[sim_indices])[::-1]]

        relevant_chunks = [ self.kb_text [i] for i in sim_indices ]
        logger.info(f"Retrieved {len((relevant_chunks))} relevant chunks")
        context = "\n\n---\n\n".join(relevant_chunks)

        return context
    
    def _cosine_sim(self, query_embed: float):
        return (
            np.dot(self.kb_embeddings, query_embed) /
            (np.linalg.norm(self.kb_embeddings, axis=1) * np.linalg.norm(query_embed))
        )
