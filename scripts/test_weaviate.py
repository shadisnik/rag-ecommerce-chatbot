import weaviate

client = weaviate.connect_to_local(port=8079 , grpc_port=50050)
print("Connected:" , client.is_ready())
client.close()