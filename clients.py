"""Qdrant / LLM 클라이언트 초기화 (앱 전체에서 공유)."""
import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient

import config


@st.cache_resource
def init_clients():
    qdrant = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        cloud_inference=True,
    )
    if not qdrant.collection_exists(config.COLLECTION):
        qdrant.create_collection(
            collection_name=config.COLLECTION,
            vectors_config=config.VECTORS_CONFIG,
        )
    llm = OpenAI(base_url=config.VLLM_URL, api_key=config.VLLM_API_KEY)
    return qdrant, llm


qdrant, llm = init_clients()
