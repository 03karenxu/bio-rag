import matplotlib.pyplot as plt
import networkx as nx
from config import FIGURES_DIR
from pyvis.network import Network
import os

def save_fig(title: str) -> None:
    plt.savefig(str(FIGURES_DIR / (title.replace(" ", "_").lower() + ".png")), bbox_inches='tight')


def visualize_graph(graph, title: str | None = None, show: bool = False) -> None:
    '''
    renders a Graph as an interactive directed subject-predicate-object network 
    and saves it as an HTML file to FIGURES_DIR.
    '''
    digraph = nx.DiGraph()
    for entity in graph.entities:
        digraph.add_node(entity)
    for triple in graph.triples:
        digraph.add_edge(triple.subject, triple.object, label=triple.predicate)

    if digraph.number_of_nodes() == 0:
        raise ValueError("Graph has no entities to visualize")

    # 1. Initialize PyVis Network (directed=True gives us arrows)
    # notebook=False ensures it saves cleanly to a standalone file
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="#000000", directed=True)

    # 2. Configure default node/edge styles to match your original colors
    net.options.nodes = {
        "color": {"background": "#4C9BE8", "border": "#1f4e79"},
        "size": 25,
        "font": {"size": 14}
    }
    net.options.edges = {
        "color": {"color": "#888888"},
        "font": {"size": 10, "color": "#444444", "align": "top"},
        "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}}
    }

    # 3. Import the NetworkX graph structure into PyVis
    net.from_nx(digraph)

    # 4. Optional: Enable physics buttons in the HTML so you can fine-tune the layout live
    # net.show_buttons(filter_=['physics'])

    # 5. Define file naming and pathing
    label = title or graph.source_doc
    filename = f"{label if title is None else title}.html"
    
    # Assuming FIGURES_DIR is a globally defined path string or Path object
    output_path = os.path.join(FIGURES_DIR, filename)

    # 6. Save and optionally launch the browser
    if show:
        net.show(output_path, local=False)
    else:
        net.write_html(output_path)