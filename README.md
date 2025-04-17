# Ragu
Rag pipeline project

### Install MongoDB Server

brew tap mongodb/brew
brew install mongodb-community@7.0

### Use Docker

docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -v mongodb_data:/data/db \
  mongo:7.0

Test it:

docker exec -it mongodb mongosh

Perform RWD:

    use testDB

    db.testCollection.insertOne({ name: "Alice", age: 30 })
    db.testCollection.find()
    db.testCollection.deleteOne({ name: "Alice" })

Check it on MongoDB Compass on the following endpoint: mongodb://localhost:27017

### Milvus

For milvus, use the docker-compose.yml to instantiate the service and the ATTU UI.
On Mac M1 Rosetta compatibility issues may arise with the pymilvus version.

Create a fresh new env for pymilvus, and do the following installation:
pip uninstall pymilvus
pip install pymilvus==2.0.0rc9

Then:

docker compose down -v
docker compose pull
docker compose up -d

Remember that on your ATTU server the name of the Milvus server is specified in the docker-compose. In this case is milvus-standalone as specified in the container name.