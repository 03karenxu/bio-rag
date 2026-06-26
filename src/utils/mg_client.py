import webbrowser
from neo4j import GraphDatabase
from grade.schema import Triple
from collections import defaultdict

class MemgraphClient:
    def __init__(self, uri: str = "bolt://localhost:7687"):
        self._driver = GraphDatabase.driver(uri, auth=("", ""))

    def close(self):
        with self._driver.session() as session:
            session.run("DROP GRAPH")
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def load_triples(self, triples: list[Triple]) -> None:
        with self._driver.session() as session:
            session.run("STORAGE MODE IN_MEMORY_ANALYTICAL")
            session.run("DROP GRAPH")
            session.run("""
                UNWIND $triples AS t
                MERGE (s:Entity {name: t.subject})
                MERGE (o:Entity {name: t.object})
                CREATE (s)-[:RELATION {
                    id: t.id,
                    predicate: t.predicate,
                    source_claim: t.source_claim,
                    source_sentence: t.source_sentence
                }]->(o)
            """, triples=[t.model_dump() for t in triples])

    def open_lab(self) -> None:
        webbrowser.open("http://localhost:3000")

    def get_shortest_paths(self) -> dict[int, list[list[Triple]]]:
        with self._driver.session() as session:
            result = session.run("""
                MATCH (start:Entity), (end:Entity)
                WHERE start <> end
                MATCH path = (start)-[:RELATION *BFS 2..5]->(end)
                WITH start, end, collect(path)[0] AS path
                RETURN path
            """)
            all_paths = []
            for record in result:
                path = record["path"]
                all_paths.append([
                    Triple(
                        id=path.relationships[i]["id"],
                        subject=path.nodes[i]["name"],
                        predicate=path.relationships[i]["predicate"],
                        object=path.nodes[i+1]["name"],
                        source_claim=path.relationships[i]["source_claim"],
                        source_sentence=path.relationships[i]["source_sentence"],
                    )
                    for i in range(len(path.relationships))
                ])

            by_hops: dict[int, list[list[Triple]]] = defaultdict(list)
            for path in all_paths:
                by_hops[len(path)].append(path)
            return by_hops
        
