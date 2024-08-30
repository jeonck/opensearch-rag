import streamlit as st
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection
import pandas as pd
import numpy as np
from tqdm import tqdm
from langchain.schema import Document  # Document 클래스를 명시적으로 임포트

# OpenSearch 클라이언트 설정
st.title("OpenSearch를 사용한 시맨틱 검색")
st.write("OpenSearch 연결 설정을 구성하세요.")

host = st.text_input("OpenSearch 호스트", "opensearch-dev.aai")  # 호스트 설정 필요
port = st.number_input("포트", value=443)
username = st.text_input("사용자 이름", "")
password = st.text_input("비밀번호", "", type="password")

client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=(username, password) if username and password else None,
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    connection_class=RequestsHttpConnection
)

# 임베딩 모델 설정
if "ko_embedding" not in st.session_state:
    model_name = "jhgan/ko-sbert-nli"
    encode_kwargs = {'normalize_embeddings': True}
    st.session_state.ko_embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )

# CSV 파일 업로드 UI
st.write("영화 데이터를 포함한 CSV 파일을 업로드하세요.")
uploaded_file = st.file_uploader("CSV 파일 선택", type="csv")

vector_index_name = "movie_semantic_vector"
keyword_index_name = "movie_keyword_index"

if uploaded_file is not None:
    # CSV 파일을 데이터프레임으로 읽기
    df = pd.read_csv(uploaded_file)
    st.write("업로드된 데이터:")
    st.dataframe(df.head())

    # 인덱스 설정 및 생성
    if st.button("OpenSearch에 데이터 인덱싱 설정"):
        # 벡터 인덱스 생성
        if client.indices.exists(index=vector_index_name):
            client.indices.delete(index=vector_index_name)

        vector_index_settings = {
            "settings": {
                "index": {
                    "knn": True  # KNN 검색 활성화
                },
                "knn.space_type": "innerproduct"  # 거리 계산을 위한 공간 유형 (Inner Product 사용)
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "plot": {"type": "text"},
                    "genre": {"type": "text"},
                    "rating": {"type": "float"},
                    "main_act": {"type": "text"},
                    "vector_field": {
                        "type": "knn_vector",
                        "dimension": 768,  # 임베딩 모델의 차원에 맞게 설정
                        "method": {
                            "name": "hnsw",  # HNSW 알고리즘 사용
                            "space_type": "innerproduct",  # 거리 계산 방법 (Inner Product 사용)
                            "engine": "faiss"  # 검색 엔진으로 FAISS 사용
                        }
                    }
                }
            }
        }

        client.indices.create(index=vector_index_name, body=vector_index_settings)
        st.write(f"벡터 인덱스 '{vector_index_name}'가 성공적으로 생성되었습니다.")

        # 키워드 인덱스 생성
        if client.indices.exists(index=keyword_index_name):
            client.indices.delete(index=keyword_index_name)

        keyword_index_settings = {
            "settings": {
                "max_result_window": 10000
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "plot": {"type": "text"},
                    "genre": {"type": "text"},
                    "rating": {"type": "float"},
                    "main_act": {"type": "text"}
                }
            }
        }

        client.indices.create(index=keyword_index_name, body=keyword_index_settings)
        st.write(f"키워드 인덱스 '{keyword_index_name}'가 성공적으로 생성되었습니다.")

        # 데이터 인덱싱
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
        documents = [
            Document(page_content=row["plot"], metadata=row.to_dict())
            for _, row in df.iterrows()
        ]
        splits = text_splitter.split_documents(documents)

        # 벡터 인덱스에 데이터 인덱싱
        st.write("### 벡터 데이터 인덱싱 중")
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, split in tqdm(enumerate(splits), total=len(splits), desc="벡터 데이터 인덱싱 중"):
            embedding = st.session_state.ko_embedding.embed_query(split.page_content)
            doc_id = hashlib.md5(split.page_content.encode('utf-8')).hexdigest()

            client.index(
                index=vector_index_name,
                id=doc_id,
                body={
                    "title": split.metadata["title"],  # Metadata에서 title 사용
                    "plot": split.page_content,
                    "genre": split.metadata["genre"],  # Metadata에서 genre 사용
                    "rating": split.metadata["rating"],  # Metadata에서 rating 사용
                    "main_act": split.metadata["main_act"],  # Metadata에서 main_act 사용
                    "vector_field": embedding
                }
            )

            progress_percentage = (i + 1) / len(splits)
            progress_bar.progress(progress_percentage)
            status_text.text(f"벡터 데이터 인덱싱 중: {i + 1}/{len(splits)} 문서")

        st.success(f"벡터 데이터가 '{vector_index_name}'에 성공적으로 인덱싱되었습니다.")

        # 키워드 인덱스에 데이터 인덱싱
        st.write("### 키워드 데이터 인덱싱 중")
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, split in tqdm(enumerate(splits), total=len(splits), desc="키워드 데이터 인덱싱 중"):
            doc_id = hashlib.md5(split.page_content.encode('utf-8')).hexdigest()

            client.index(
                index=keyword_index_name,
                id=doc_id,
                body={
                    "title": split.metadata["title"],  # Metadata에서 title 사용
                    "plot": split.page_content,
                    "genre": split.metadata["genre"],  # Metadata에서 genre 사용
                    "rating": split.metadata["rating"],  # Metadata에서 rating 사용
                    "main_act": split.metadata["main_act"]  # Metadata에서 main_act 사용
                }
            )

            progress_percentage = (i + 1) / len(splits)
            progress_bar.progress(progress_percentage)
            status_text.text(f"키워드 데이터 인덱싱 중: {i + 1}/{len(splits)} 문서")

        st.success(f"키워드 데이터가 '{keyword_index_name}'에 성공적으로 인덱싱되었습니다.")


# KNN 검색 함수
def knn_search(client, index_name, query_vector, k=10):
    query = {
        "size": k,
        "query": {
            "knn": {
                "vector_field": {
                    "vector": query_vector,
                    "k": k
                }
            }
        }
    }

    res = client.search(index=index_name, body=query)

    query_result = []
    for hit in res["hits"]["hits"]:
        row = [
            hit["_score"],
            hit["_source"]["title"],
            hit["_source"]["plot"],
            hit["_source"]["genre"],
            hit["_source"]["rating"],
            hit["_source"]["main_act"],
        ]
        query_result.append(row)

    query_result_df = pd.DataFrame(
        data=query_result, columns=["_score", "title", "plot", "genre", "rating", "main_act"]
    )
    return query_result_df


# Multi_match 검색 함수
def keyword_search(client, index_name, query_text, k=10):
    query = {
        "size": k,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": ["title", "plot", "genre", "main_act"],
                "type": "best_fields"
            }
        }
    }

    res = client.search(index=index_name, body=query)

    query_result = []
    for hit in res["hits"]["hits"]:
        row = [
            hit["_score"],
            hit["_source"]["title"],
            hit["_source"]["plot"],
            hit["_source"]["genre"],
            hit["_source"]["rating"],
            hit["_source"]["main_act"],
        ]
        query_result.append(row)

    query_result_df = pd.DataFrame(
        data=query_result, columns=["_score", "title", "plot", "genre", "rating", "main_act"]
    )
    return query_result_df


# RRF 기반 하이브리드 검색 함수
def hybrid_search(client, keyword_index_name, vector_index_name, query_text, query_vector, keyword_weight=0.3, semantic_weight=1.0, k=10, rrf_const=60):
    # 키워드 검색 결과
    keyword_results = keyword_search(client, keyword_index_name, query_text, k)
    keyword_results["_rank"] = range(1, len(keyword_results) + 1)
    keyword_results["_score"] = keyword_results["_score"] * keyword_weight / (rrf_const + keyword_results["_rank"])

    # 시맨틱 검색 결과
    semantic_results = knn_search(client, vector_index_name, query_vector, k)
    semantic_results["_rank"] = range(1, len(semantic_results) + 1)
    semantic_results["_score"] = semantic_results["_score"] * semantic_weight / (rrf_const + semantic_results["_rank"])

    # 두 결과를 결합
    combined_results = pd.concat([keyword_results, semantic_results]).groupby("title").agg({
        "_score": "sum",
        "plot": "first",
        "genre": "first",
        "rating": "first",
        "main_act": "first",
        "_rank": "min"
    }).sort_values(by=["_score", "_rank"], ascending=[False, True])

    return combined_results.head(k)


# Streamlit 검색 UI
st.write("영화를 검색할 쿼리를 입력하세요.")
query_text = st.text_input("검색 쿼리 입력:", "우주에서 벌어지는 전쟁 이야기를 다룬 영화 소개해줘")

if st.button("검색"):
    if query_text:
        # KNN 검색 (시맨틱 검색)
        st.write("시맨틱 검색(KNN 검색) 결과:")
        query_vector = st.session_state.ko_embedding.embed_query(query_text)
        knn_results = knn_search(client, vector_index_name, query_vector)
        st.dataframe(knn_results)

        # Multi_match 검색 (키워드 검색)
        st.write("키워드 검색 결과:")
        keyword_results = keyword_search(client, keyword_index_name, query_text)
        st.dataframe(keyword_results)

        # RRF 기반 하이브리드 검색
        st.write("RRF 하이브리드 검색 결과:")
        hybrid_results = hybrid_search(client, keyword_index_name, vector_index_name, query_text, query_vector)
        st.dataframe(hybrid_results)
