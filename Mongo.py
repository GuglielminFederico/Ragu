import yaml
from pymongo import MongoClient

class MongoDB:

    def __init__(self, config_path):

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.host = self.config.get("host", None)
        self.port = self.config.get("port", None)
        self.db_name = self.config.get("db_name", None)
        self.client = MongoClient(self.host, self.port)
        self.db = self.client[self.db_name]

    def insert_one(self, collection_name, document):
        collection = self.db[collection_name]
        result = collection.insert_one(document)
        return result.inserted_id

    def find_one(self, collection_name, query):
        collection = self.db[collection_name]
        document = collection.find_one(query)
        return document

    def update_one(self, collection_name, query, update):
        collection = self.db[collection_name]
        result = collection.update_one(query, update)
        return result.modified_count

    def delete_one(self, collection_name, query):
        collection = self.db[collection_name]
        result = collection.delete_one(query)
        return result.deleted_count 
    
    def insert_many(self, collection_name, documents):
        collection = self.db[collection_name]
        result = collection.insert_many(documents)
        return result.inserted_ids