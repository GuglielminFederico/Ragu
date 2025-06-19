
from pymilvus import connections, utility, CollectionSchema, FieldSchema, DataType, Collection

class Milvus:
    def __init__(self, host="localhost", port="19530"):
        self.host = host
        self.port = port
        self.collection_name = "rag"
        self.collection = None

    def connect(self):
        connections.connect("default", host=self.host, port=self.port)

    def create_collection(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4)
        ]
        schema = CollectionSchema(fields, description="Example schema")

        self.collection = Collection(name=self.collection_name, schema=schema)

    def insert_data(self, ids, embeddings):
        if not self.collection:
            raise Exception("Collection not created. Call create_collection() first.")
        
        self.collection.insert([ids, embeddings])



# 1. Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# 2. Define collection and schema
collection_name = "test"

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4)
]
schema = CollectionSchema(fields, description="Example schema")

collection = Collection(name=collection_name, schema=schema)

# 3. Insert data
ids = [1]
embeddings = [[random.random() for _ in range(4)]]
collection.insert([ids, embeddings])