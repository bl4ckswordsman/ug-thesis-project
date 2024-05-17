import pydot
from IPython.display import Image, display


def display_graph(dot_file_path):
    # Load the graph from the .dot file
    graph = pydot.graph_from_dot_file(dot_file_path)

    # Create a PNG image from the graph
    png_image = graph[0].create(format='png')

    # Display the image
    display(Image(png_image))


