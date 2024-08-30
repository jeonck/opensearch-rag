# opensearch-rag

코드내 알고리즘 설명
ANN(Approximate Nearest Neighbor)은 대규모 데이터셋에서 특정 벡터와 유사한 벡터를 빠르게 찾기 위해 사용되는 기술입니다. 특히, 벡터 공간에서의 검색은 고차원 데이터(예: 텍스트 임베딩)에서 매우 중요한데, 정확한 최근접 이웃을 찾는 데 시간이 많이 소요되기 때문에 ANN 알고리즘을 통해 이를 근사치로 빠르게 찾을 수 있습니다.

코드에서의 ANN 사용
HNSW 알고리즘: 코드에서 벡터 인덱스를 설정할 때, knn_vector 필드에 HNSW(Hierarchical Navigable Small World) 알고리즘을 사용하고 있습니다. 이 알고리즘은 그래프 기반의 ANN 방식으로, 고차원 벡터 공간에서 효율적인 근사 최근접 이웃 검색을 제공합니다.

python

"method": {
    "name": "hnsw",  # HNSW 알고리즘 사용
    "space_type": "innerproduct",  # 거리 계산 방법 (Inner Product 사용)
    "engine": "faiss"  # 검색 엔진으로 FAISS 사용
}
FAISS 엔진: 검색 엔진으로 Facebook AI에서 개발한 FAISS를 사용하고 있습니다. FAISS는 대규모 벡터 검색 및 최근접 이웃 검색에 특화된 라이브러리로, 특히 HNSW와 같은 ANN 알고리즘을 지원하여 효율적인 검색을 가능하게 합니다.

KNN 검색에서의 사용: KNN 검색 함수에서 knn_vector 필드를 사용하여 벡터 검색을 수행하고 있으며, 이 과정에서 ANN 방식을 통해 빠르게 유사한 벡터를 찾습니다.

python

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
요약
이 예제에서 ANN 방식은 HNSW 알고리즘과 FAISS 엔진을 통해 대규모 벡터 공간에서 효율적인 유사도 검색을 수행하는 데 사용됩니다. ANN을 활용함으로써, 정확한 유사도 검색은 다소 포기하더라도, 실시간 검색을 가능하게 하여 실용성을 높이는 것이 목적입니다.
