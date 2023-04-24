import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import cascaded_union
import random
import math
import networkx as nx
import streamlit as st
from PIL import Image

# # Create sidebar to enter the number of houses, hills, valleys, and building labels
# no_of_houses = 20 # slider, default 20 , min 2, max 100
# no_of_valleys = 2 # slider, default 2 , min 0, max 10
# no_of_hills = 3 # slider , default 3 , min 0, max 10
# house_width, house_height = 20, 20 # number , default 20
# hill_radius = 30 # number , default 30
# valley_radius = 20 # number , default 20
# building_labels = ["Hospital", "Gas Station", "Public Park", "School"]
# vertex_weights = {"House": 1, "Hospital": 2, "Gas Station": 3, "Public Park": 4, "School": 5}


def main():
    st.title("Navigating the Urban Jungle : Optimizing Road Network")

    st.sidebar.header("Input Parameters")
    no_of_houses = st.sidebar.slider("Number of houses", 2, 50, 20)
    no_of_valleys = st.sidebar.slider("Number of valleys", 1, 10, 2)
    no_of_hills = st.sidebar.slider("Number of hills", 1, 10, 3)
    house_width = st.sidebar.number_input("House width", value=20)
    house_height = st.sidebar.number_input("House height", value=20)
    hill_radius = st.sidebar.number_input("Hill radius", value=30)
    valley_radius = st.sidebar.number_input("Valley radius", value=20)


    # Building labels and vertex weights input
    # building_labels_input = st.sidebar.text_input("Building Labels (comma separated)", value="Hospital, Gas Station, Public Park, School")
    # building_labels = [label.strip() for label in building_labels_input.split(',')]
    # vertex_weights_input = st.sidebar.text_input("Vertex Weights (label:weight, comma separated)", value="House:1, Hospital:2, Gas Station:3, Public Park:4, School:5")
    # vertex_weights = {pair.split(':')[0].strip(): int(pair.split(':')[1].strip()) for pair in vertex_weights_input.split(',')}
    # num_buildings_input = st.sidebar.text_input("Number of Buildings (label:number, comma separated)", value="Hospital:1, Gas Station:1, Public Park:1, School:1")
    # num_buildings = {pair.split(':')[0].strip(): int(pair.split(':')[1].strip()) for pair in num_buildings_input.split(',')}

    image_with_vertex_weights, image_with_mst, image_with_tsp, image_with_tsp_2opt,mst_total_weight,combined_weight_2_opt,combined_weight_tsp = generate_optimal_road_network(
        no_of_houses=no_of_houses,
        no_of_valleys=no_of_valleys,
        no_of_hills=no_of_hills,
        house_width=house_width,
        house_height=house_height,
        hill_radius=hill_radius,
        valley_radius=valley_radius,
    )

    st.header("Connected Graph")
    st.image(image_with_vertex_weights, caption="Connected graph with vertex and edge weights", use_column_width=True)

    st.header("Minimum Spanning Tree and TSP Solutions")
    col1, col2, col3 = st.columns(3)
    col1.image(image_with_mst, caption= f"Minimum Spanning Tree", use_column_width=True)
    # col1.write(f"weight: {round(mst_total_weight,2)}")
    col1.markdown(f"<h3 style='text-align: center;'>weight : {round(mst_total_weight,2)}</h3>", unsafe_allow_html=True)
    col2.image(image_with_tsp, caption=f"Nearest Neighbor TSP Solution", use_column_width=True)
    # col2.write(f"weight: {round(combined_weight_tsp,2)}")
    col2.markdown(f"<h3 style='text-align: center;'>weight : {round(combined_weight_tsp,2)}</h3>", unsafe_allow_html=True)
    col3.image(image_with_tsp_2opt, caption=f"2-Opt TSP Solution", use_column_width=True)
    # col3.write(f"weight: {round(combined_weight_2_opt,2)}")
    col3.markdown(f"<h3 style='text-align: center;'>weight : {round(combined_weight_2_opt,2)}</h3>", unsafe_allow_html=True)


def generate_optimal_road_network(
    no_of_houses,
    no_of_valleys,
    no_of_hills,
    house_width,
    house_height,
    hill_radius,
    valley_radius,
):

    building_labels = ["Hospital", "Gas Station", "Public Park", "School"]
    vertex_weights = {"House": 1, "Hospital": 2, "Gas Station": 3, "Public Park": 4, "School": 5}
    building_width, building_height = random.randint(30, 60), random.randint(30, 60) # no input needed, use as is

    def is_complete(graph: nx.Graph) -> bool:
        n = len(graph.nodes)
        max_edges = (n * (n - 1)) // 2
        return len(graph.edges) == max_edges

    def random_non_intersecting_polygon(existing_polygons, width, height):
        while True:
            x, y = random.randint(0, 512 - width), random.randint(0, 512 - height)
            new_polygon = Polygon([(x, y), (x + width, y), (x + width, y + height), (x, y + height)])
            
            intersects = False
            for existing_polygon in existing_polygons:
                if new_polygon.intersects(existing_polygon):
                    intersects = True
                    break
            
            if not intersects:
                return new_polygon

    # Initialize the image
    image = np.zeros((512, 512), dtype=np.uint8)

    # Draw a hill
    for _ in range(no_of_hills):
        hill_center = (random.randint(hill_radius, 512 - hill_radius), random.randint(hill_radius, 512 - hill_radius))
        cv2.circle(image, hill_center, hill_radius, 255, -1)
        hill_polygon = Polygon([(hill_center[0] + hill_radius * np.cos(angle), hill_center[1] + hill_radius * np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, 100)]).buffer(0)

    # Draw valleys
    valley_polygons = []
    for _ in range(no_of_valleys):
        x, y = random.randint(0, 512 - 1), random.randint(0, 512 - 1)
        valley_polygon = Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]).buffer(valley_radius)
        valley_polygons.append(valley_polygon)

        for x, y in valley_polygon.exterior.coords:
            cv2.circle(image, (int(x), int(y)), valley_radius, 255, -1)

    # Draw houses
    house_polygons = []
    for _ in range(no_of_houses):
        house_polygon = random_non_intersecting_polygon(house_polygons + valley_polygons + [hill_polygon], house_width, house_height)
        house_polygons.append(house_polygon)

        x, y = int(house_polygon.bounds[0]), int(house_polygon.bounds[1])
        cv2.rectangle(image, (x, y), (x + house_width, y + house_height), 255, -1)

    # Draw buildings with labels
    label_polygons = []
    for label in building_labels:
        building_polygon = random_non_intersecting_polygon(house_polygons + valley_polygons + [hill_polygon], building_width, building_height)
        label_polygons.append((building_polygon, label))

        x, y = int(building_polygon.bounds[0]), int(building_polygon.bounds[1])
        cv2.rectangle(image, (x, y), (x + building_width, y + building_height), 255, -1)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

    def get_polygon_center(polygon):
        x, y = np.array(polygon.centroid.coords).squeeze()
        return int(x), int(y)

    #Extract center points of shapes
    center_points = []
    for polygon in house_polygons:
        center_points.append(get_polygon_center(polygon))

    # Add center points for labeled buildings
    for polygon, label in label_polygons:
        center_points.append(get_polygon_center(polygon))

    # Create a complete graph using the NetworkX library
    G = nx.complete_graph(len(center_points))

    # Set the node positions to the center points of the shapes
    pos = {i: center_points[i] for i in range(len(center_points))}
    nx.set_node_attributes(G, pos, "pos")

    # Visualize the graph
    image_with_graph = image.copy()
    for i, j in G.edges():
        cv2.line(image_with_graph, center_points[i], center_points[j], 128, 1)

    # Check if a line segment intersects with a polygon
    def line_intersects_polygon(line_start, line_end, polygon):
        line = LineString([line_start, line_end])
        return line.intersects(polygon)

    # Calculate Euclidean distance between two points
    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def edge_weight(p1, p2, hill_polygon, valley_polygons):
        weight = euclidean_distance(p1, p2)
        line = LineString([p1, p2])
        
        if line.intersects(hill_polygon):
            weight *= 2
        
        if any(line.intersects(valley) for valley in valley_polygons):
            weight *= 3

        return weight

    # Add edge weights based on the distance and the presence of hills or valleys
    for i, j in G.edges():
        G[i][j]['weight'] = edge_weight(center_points[i], center_points[j], hill_polygon, valley_polygons)

    # Visualize the graph with edge weights
    image_with_graph_weights = image.copy()
    for i, j in G.edges():
        cv2.line(image_with_graph_weights, center_points[i], center_points[j], 128, 1)
        
        # Calculate the midpoint of the edge
        midpoint = ((center_points[i][0] + center_points[j][0]) // 2, (center_points[i][1] + center_points[j][1]) // 2)
        
        # Draw the edge weight as a text label
        weight = int(G[i][j]['weight'])
        cv2.putText(image_with_graph_weights, str(weight), midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        
    # Set the node positions to the center points of the shapes
    pos = {i: center_points[i] for i in range(len(center_points))}
    nx.set_node_attributes(G, pos, "pos")

    # Assign vertex weights based on the type of building
    vertex_weights_list = [vertex_weights["House"]] * no_of_houses
    for _, label in label_polygons:
        vertex_weights_list.append(vertex_weights[label])
    nx.set_node_attributes(G, {i: vertex_weights_list[i] for i in range(len(center_points))}, "weight")

    # Visualize the graph with vertex weights
    image_with_vertex_weights = image_with_graph_weights.copy()
    for i in range(len(center_points)):
        x, y = center_points[i]
        weight = int(G.nodes[i]['weight'])
        cv2.putText(image_with_vertex_weights, str(weight), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

    print(is_complete(G))

    # Display image image_with_vertex_weights

    def kruskal_with_vertex_weights(graph):
        sorted_edges = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'] + graph.nodes[x[0]]['weight'] + graph.nodes[x[1]]['weight'])
        mst = nx.Graph()

        for u, v, data in sorted_edges:
            if u not in mst or v not in mst or not nx.has_path(mst, u, v):
                mst.add_edge(u, v, weight=data['weight'])

        return mst

    # Find the Minimum Spanning Tree
    mst = kruskal_with_vertex_weights(G)

    # Visualize the MST
    image_with_mst = image.copy()
    for i, j in mst.edges():
        cv2.line(image_with_mst, center_points[i], center_points[j], 128, 2)

        # Calculate the midpoint of the edge
        midpoint = ((center_points[i][0] + center_points[j][0]) // 2, (center_points[i][1] + center_points[j][1]) // 2)
        
        # Draw the edge weight as a white text label
        weight = int(mst[i][j]['weight'])
        cv2.putText(image_with_mst, str(weight), midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

    for i in range(len(center_points)):
        x, y = center_points[i]
        
        # Draw the vertex weight as a black text label
        weight = int(G.nodes[i]['weight'])
        cv2.putText(image_with_mst, str(weight), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

    # Display image image_with_mst

    def nearest_neighbor_algorithm(graph, start_node=None):
        if start_node is None:
            start_node = random.choice(list(graph.nodes()))

        unvisited = list(graph.nodes())
        unvisited.remove(start_node)
        tour = [start_node]

        current_node = start_node
        while unvisited:
            min_distance = float('inf')
            closest_node = None

            for node in unvisited:
                distance = graph[current_node][node]['weight']
                if distance < min_distance:
                    min_distance = distance
                    closest_node = node

            tour.append(closest_node)
            unvisited.remove(closest_node)
            current_node = closest_node

        # Connect the last node back to the start_node
        tour.append(start_node)

        return tour

    # Find the TSP tour
    tsp_tour = nearest_neighbor_algorithm(G)

    # Visualize the TSP tour
    image_with_tsp = image.copy()
    for i in range(len(tsp_tour) - 1):
        node1 = tsp_tour[i]
        node2 = tsp_tour[i + 1]
        cv2.line(image_with_tsp, center_points[node1], center_points[node2], 128, 2)

    # Display image image_with_tsp

    def two_opt_algorithm(graph, initial_tour=None):
        nodes = list(graph.nodes)
        
        def tour_length(tour):
            return sum(graph[tour[i - 1]][tour[i]]['weight'] for i in range(len(tour)))

        def two_opt_swap(tour, i, j):
            return tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]

        if initial_tour is None:
            initial_tour = nodes

        current_tour = initial_tour
        current_length = tour_length(current_tour)

        improved = True
        while improved:
            improved = False

            for i in range(1, len(current_tour) - 2):
                for j in range(i + 1, len(current_tour) - 1):
                    new_tour = two_opt_swap(current_tour, i, j)
                    new_length = tour_length(new_tour)

                    if new_length < current_length:
                        current_tour = new_tour
                        current_length = new_length
                        improved = True

        return current_tour

    # Find the TSP tour using the 2-Opt algorithm
    tsp_tour_2opt = two_opt_algorithm(G)

    # Visualize the 2-Opt TSP tour
    image_with_tsp_2opt = image.copy()
    for i in range(len(tsp_tour_2opt) - 1):
        node1 = tsp_tour_2opt[i]
        node2 = tsp_tour_2opt[i + 1]
        cv2.line(image_with_tsp_2opt, center_points[node1], center_points[node2], 128, 2)

    # Display image image_with_tsp_2opt

    # cv2.imshow("TSP Tour", image_with_tsp)
    # cv2.imshow("Minimum Spanning Tree", image_with_mst)
    # cv2.imshow("Graph with Vertex Weights", image_with_vertex_weights)
    # cv2.imshow("2-Opt TSP Tour", image_with_tsp_2opt)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    def tour_weight(tour, graph):
        weight = 0
        for i in range(len(tour) - 1):
            weight += graph[tour[i]][tour[i + 1]]['weight']
        return weight

    def mst_weight(mst):
        return sum(weight['weight'] for u, v, weight in mst.edges(data=True))

    # include under each respective image display
    mst_total_weight = mst_weight(mst)
    print("Combined weight of the Minimum Spanning Tree:", mst_total_weight)

    combined_weight_2_opt = tour_weight(tsp_tour_2opt, G)
    print("Combined weight of the optimal road network using 2-Opt TSP:", combined_weight_2_opt)

    combined_weight_tsp = tour_weight(tsp_tour, G)
    print("Combined weight of the optimal road network using NN TSP:", combined_weight_tsp)

    image_with_vertex_weights = cv2.cvtColor(image_with_vertex_weights, cv2.COLOR_GRAY2RGB)
    image_with_vertex_weights = Image.fromarray(image_with_vertex_weights)

    image_with_mst = cv2.cvtColor(image_with_mst, cv2.COLOR_GRAY2RGB)
    image_with_mst = Image.fromarray(image_with_mst)

    image_with_tsp = cv2.cvtColor(image_with_tsp, cv2.COLOR_GRAY2RGB)
    image_with_tsp = Image.fromarray(image_with_tsp)

    image_with_tsp_2opt = cv2.cvtColor(image_with_tsp_2opt, cv2.COLOR_GRAY2RGB)
    image_with_tsp_2opt = Image.fromarray(image_with_tsp_2opt)

    return image_with_vertex_weights, image_with_mst, image_with_tsp, image_with_tsp_2opt,mst_total_weight,combined_weight_2_opt,combined_weight_tsp

if __name__ == "__main__":
    main()
