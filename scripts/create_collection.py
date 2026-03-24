import weaviate
from weaviate.classes.config import Configure, Property, DataType

client = weaviate.connect_to_local(port=8079, grpc_port=50050)

try:
    if client.collections.exists("StoreDocs"):
        client.collections.delete("StoreDocs")

    client.collections.create(
        name="StoreDocs",
        vector_config=Configure.Vectors.text2vec_transformers(
            name="default"
        ),
        reranker_config=Configure.Reranker.transformers(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="doc_type", data_type=DataType.TEXT),
            Property(name="product_name", data_type=DataType.TEXT),
            Property(name="gender", data_type=DataType.TEXT),
            Property(name="category", data_type=DataType.TEXT),
            Property(name="sub_category", data_type=DataType.TEXT),
            Property(name="article_type", data_type=DataType.TEXT),
            Property(name="color", data_type=DataType.TEXT),
            Property(name="season", data_type=DataType.TEXT),
            Property(name="year", data_type=DataType.NUMBER),
            Property(name="usage", data_type=DataType.TEXT),
            Property(name="image_path", data_type=DataType.TEXT),
        ]
    )

    print("Collection 'StoreDocs' created successfully!")

finally:
    client.close()