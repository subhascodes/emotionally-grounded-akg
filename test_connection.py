from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
USERNAME = "neo4j"
PASSWORD = "namasteindia"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

with driver.session() as session:
    result = session.run("RETURN 'AKG Connected Successfully' AS message")
    print(result.single()["message"])

driver.close()