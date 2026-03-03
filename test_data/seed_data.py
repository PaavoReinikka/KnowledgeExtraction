import psycopg2
import numpy as np
from faker import Faker
import uuid
import json

# Connection parameters - adjust as needed or use environment variables
DB_PARAMS = {
    "host": "localhost",
    "port": 5433,
    "database": "postgres",
    "user": "postgres",
    "password": "mypassword"
}

fake = Faker()

def seed_documents(cursor, n=10):
    print(f"Seeding {n} documents...")
    for _ in range(n):
        content = fake.paragraph(nb_sentences=5)
        # Generating random 384-dimensional embedding for testing
        # Convert to standard Python float list to avoid numpy-specific type issues in the query
        embedding = [float(x) for x in np.random.uniform(-1, 1, 384)]
        metadata = json.dumps({"source": fake.file_name(), "author": fake.name()})
        
        cursor.execute(
            "INSERT INTO documents (content, embedding, metadata) VALUES (%s, %s, %s)",
            (content, embedding, metadata)
        )

def seed_graph(cursor):
    print("Seeding graph data with Apache AGE...")
    # 1. Create a graph if it doesn't exist
    cursor.execute("SELECT * FROM ag_catalog.create_graph('knowledge_graph');")
    
    # 2. Add some nodes (Entities)
    entities = ["Company", "Person", "Project", "Technology"]
    for entity in entities:
        name = fake.company() if entity == "Company" else fake.name()
        query = f"""
        SELECT * FROM cypher('knowledge_graph', $$
            CREATE (n:{entity} {{name: '{name}', type: '{entity}'}})
        $$) as (v agtype);
        """
        cursor.execute(query)
    
    # 3. Add some edges (Relationships)
    cursor.execute("""
    SELECT * FROM cypher('knowledge_graph', $$
        MATCH (a), (b)
        WHERE a.type = 'Person' AND b.type = 'Company'
        CREATE (a)-[r:WORKS_AT]->(b)
        RETURN r
    $$) as (e agtype);
    """)

def main():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        conn.autocommit = True
        with conn.cursor() as cur:
            # First, ensure extensions are ready (already handled by migrations, but good to check)
            cur.execute("SET search_path = public, ag_catalog;")
            
            seed_documents(cur, n=15)
            seed_graph(cur)
            
            print("Successfully seeded testing data!")
            
    except Exception as e:
        print(f"Error seeding data: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
