import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import cascaded_union
import random
import math
import networkx as nx
import itertools
import streamlit as st
from PIL import Image


# General Functions
def assign_vertex_weights(graph, house_polygons, label_polygons,vertex_weights):
    for i, _ in enumerate(house_polygons):
        graph.nodes[i]['weight'] = vertex_weights['House']
    
    for j, (_, label) in enumerate(label_polygons, start=len(house_polygons)):
        graph.nodes[j]['weight'] = vertex_weights[label]

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
        
def draw_circular_obstacle(image,number,radius):
    polygons = []
    for _ in range(number):
        center = (random.randint(radius, 512 - radius), random.randint(radius, 512 - radius))
        cv2.circle(image, center, radius, 255, -1)
        polygon = Polygon([(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, 100)]).buffer(0)
        polygons.append(polygon)
    return image,polygons


def draw_houses(image,no_of_houses,valley_polygons,hill_polygons,house_width, house_height):
    # Draw houses
    house_polygons = []
    for _ in range(no_of_houses):
        house_polygon = random_non_intersecting_polygon(house_polygons + valley_polygons + hill_polygons, house_width, house_height)
        house_polygons.append(house_polygon)

        x, y = int(house_polygon.bounds[0]), int(house_polygon.bounds[1])
        cv2.rectangle(image, (x, y), (x + house_width, y + house_height), 255, -1)
    return image,house_polygons

def draw_buildings(image,valley_polygons,hill_polygons,house_polygons,building_labels,building_width, building_height):
    # Draw buildings with labels
    label_polygons = []
    building_polygon = None
    for label in building_labels:
        building_polygon = random_non_intersecting_polygon(house_polygons + valley_polygons + hill_polygons, building_width, building_height)
        label_polygons.append((building_polygon, label))

        x, y = int(building_polygon.bounds[0]), int(building_polygon.bounds[1])
        cv2.rectangle(image, (x, y), (x + building_width, y + building_height), 255, -1)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

    return image,label_polygons,building_polygon
        
def draw_tsp_route(image,tsp,center_points,rectilinear = True):
    # Draw the rectilinear edges in a thick white line
    for i in range(len(tsp) - 1):
        node1 = tsp[i]
        node2 = tsp[i + 1]
        if rectilinear:
            draw_rectilinear_line(image, center_points[node1], center_points[node2], 255, 4)
        else:
            cv2.line(image, center_points[node1], center_points[node2], 255, 4)
    return image


def draw_mst_route(image, mst, center_points, rectilinear = True):
    for i, j in mst.edges():
        if rectilinear:
            draw_rectilinear_line(image, center_points[i], center_points[j], 255, 4)
        else:
            cv2.line(image, center_points[i], center_points[j], 255, 4)
    return image

# Create Graph and Image

def create_graph(grid_cell_size,no_of_hills,hill_radius,no_of_valleys,valley_radius,no_of_houses,house_width, house_height,building_labels,building_width, building_height,vertex_weights,rectilinear):
    # Initialize the image
    image = np.zeros((512, 512), dtype=np.uint8)

    if rectilinear:
        image = draw_grid(image, grid_cell_size)

    # Draw hill
    image,hill_polygons = draw_circular_obstacle(image,no_of_hills,hill_radius)

    # Draw valleys
    image,valley_polygons = draw_circular_obstacle(image,no_of_valleys,valley_radius)

    # Draw houses
    image,house_polygons = draw_houses(image,no_of_houses,valley_polygons,hill_polygons,house_width, house_height)

    # Draw buildings with labels
    image,label_polygons,building_polygon = draw_buildings(image,valley_polygons,hill_polygons,house_polygons,building_labels,building_width, building_height)

    #Extract center points of shapes
    center_points = []
    for polygon in house_polygons:
        if rectilinear:
            center_points.append(get_polygon_center_rectilinear(polygon,grid_cell_size))
        else:
            center_points.append(get_polygon_center(polygon))

    # Add center points for labeled buildings
    for polygon, label in label_polygons:
        if rectilinear:
            center_points.append(get_polygon_center_rectilinear(polygon,grid_cell_size))
        else:
            center_points.append(get_polygon_center(polygon))

    # Create a complete graph using the NetworkX library
    G = nx.complete_graph(len(center_points))

    # Set the node positions to the center points of the shapes
    pos = {i: center_points[i] for i in range(len(center_points))}
    nx.set_node_attributes(G, pos, "pos")

    # Assign weights to the vertexes
    assign_vertex_weights(G, house_polygons, label_polygons,vertex_weights)

    # Assign weights to the edges
    for i, j in G.edges():
        if rectilinear:
            G[i][j]['weight'] = edge_weight_with_rectilinear(center_points[i], center_points[j], hill_polygons, valley_polygons)
        else:
            G[i][j]['weight'] = edge_weight_with_euclidean(center_points[i], center_points[j], hill_polygons, valley_polygons)


    # Visualize the graph with edge weights
    for i, j in G.edges():

        if rectilinear:
            draw_rectilinear_line(image, center_points[i], center_points[j], 255, 1)
        else:
            cv2.line(image, center_points[i], center_points[j], 255, 1)

    # Draw the vertex weights as black text labels
    for i in range(len(center_points)):
        x, y = center_points[i]
        weight = int(G.nodes[i]['weight'])
        cv2.putText(image, str(weight), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

    image_with_graph_weights = image.copy() 

    for i, j in G.edges():

        # Calculate the midpoint of the edge
        midpoint = ((center_points[i][0] + center_points[j][0]) // 2, (center_points[i][1] + center_points[j][1]) // 2)
        
        # Draw the edge weight as a text label
        weight = int(G[i][j]['weight'])
        cv2.putText(image_with_graph_weights, str(weight), midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1)

    return G,image_with_graph_weights,image,center_points

# Routing Algorithms

def brute_force_tsp(graph):
    nodes = list(graph.nodes())
    min_length = float('inf')
    best_tour = None

    # Iterate over all possible permutations of nodes
    for tour in itertools.permutations(nodes[1:]):
        tour = [nodes[0]] + list(tour) + [nodes[0]]
        length = sum(graph[tour[i]][tour[i + 1]]['weight'] for i in range(len(tour) - 1))

        if length < min_length:
            min_length = length
            best_tour = tour

    return best_tour, min_length


def nearest_neighbor_tsp_algorithm(graph, start_node=None):
    def tour_length(tour):
        return sum(graph[tour[i]][tour[i + 1]]['weight'] for i in range(len(tour) - 1))

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

    min_length = tour_length(tour)
    return tour, min_length


# def two_opt_algorithm(graph, initial_tour=None):
#     nodes = list(graph.nodes)
    
#     def tour_length(tour):
#         return sum(graph[tour[i - 1]][tour[i]]['weight'] for i in range(len(tour)))

#     def two_opt_swap(tour, i, j):
#         return tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]

#     if initial_tour is None:
#         initial_tour = nodes

#     current_tour = initial_tour
#     current_length = tour_length(current_tour)

#     improved = True
#     while improved:
#         improved = False

#         for i in range(1, len(current_tour) - 2):
#             for j in range(i + 1, len(current_tour) - 1):
#                 new_tour = two_opt_swap(current_tour, i, j)
#                 new_length = tour_length(new_tour)

#                 if new_length < current_length:
#                     current_tour = new_tour
#                     current_length = new_length
#                     improved = True
#     min_length = current_length
#     return current_tour, min_length



def two_opt_algorithm(graph, initial_tour=None):
    nodes = list(graph.nodes)

    def tour_length(tour):
        total_weight = 0
        for i in range(len(tour) - 1):
            if graph.has_edge(tour[i], tour[i + 1]):
                total_weight += graph[tour[i]][tour[i + 1]]['weight']
        return total_weight

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

    # Connect the last node back to the start_node
    current_tour.append(current_tour[0])

    # Calculate the min_length considering the connection back to the start_node
    min_length = tour_length(current_tour)
    return current_tour, min_length




def kruskal_with_vertex_weights(graph):
    sorted_edges = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'] - graph.nodes[x[0]]['weight'] - graph.nodes[x[1]]['weight'])
    mst = nx.Graph()
    min_length = 0

    for u, v, data in sorted_edges:
        if u not in mst or v not in mst or not nx.has_path(mst, u, v):
            mst.add_edge(u, v, weight=data['weight'])
            min_length += data['weight']

    return mst, min_length


# Functions for Rectilinear
def get_polygon_center_rectilinear(polygon,grid_cell_size):
    x, y = np.array(polygon.centroid.coords).squeeze()
    x = round(x / grid_cell_size) * grid_cell_size
    y = round(y / grid_cell_size) * grid_cell_size
    return int(x), int(y)


# Draw rectilinear lines
def draw_rectilinear_line(image, p1, p2, color, thickness):
    x1, y1 = p1
    x2, y2 = p2
    cv2.line(image, (x1, y1), (x1, y2), color, thickness)
    cv2.line(image, (x1, y2), (x2, y2), color, thickness)


# Calculate rectilinear distance between two points
def rectilinear_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# Calculate weight of an edge between two points
def edge_weight_with_rectilinear(p1, p2, hill_polygons, valley_polygons):
    weight = rectilinear_distance(p1, p2)
    line1 = LineString([p1, (p1[0], p2[1])])
    line2 = LineString([(p1[0], p2[1]), p2])
    
    if any(line1.intersects(hill) or line2.intersects(hill) for hill in hill_polygons):
        weight *= 2
    
    if any(line1.intersects(valley) or line2.intersects(valley) for valley in valley_polygons):
        weight *= 3

    return weight

def draw_grid(image,grid_cell_size):
    # Draw the grid lines in a thin line
    for x in range(0, 512, grid_cell_size):
        cv2.line(image, (x, 0), (x, 512), 30, 1)
        cv2.line(image, (0, x), (512, x), 30, 1)
    return image


# Functions for Euclidean

def get_polygon_center(polygon):
    x, y = np.array(polygon.centroid.coords).squeeze()
    return int(x), int(y)

# Calculate Euclidean distance between two points
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Calculate weight of an edge between two points
def edge_weight_with_euclidean(p1, p2, hill_polygons, valley_polygons):
    weight = euclidean_distance(p1, p2)
    line = LineString([p1, p2])
    
    if any(line.intersects(hill) for hill in hill_polygons):
        weight *= 2
    
    if any(line.intersects(valley) for valley in valley_polygons):
        weight *= 3

    return weight


def generate_optimal_road_network(rectilinear,grid_cell_size,house_width,house_height,hill_radius,valley_radius,building_width,building_height,compute_capacity,building_labels,vertex_weights,no_of_houses,no_of_valleys,no_of_hills,no_of_buildings):
    
    # Create Complete Graph
    G, image_with_graph_weights, image ,center_points = create_graph(grid_cell_size,no_of_hills,hill_radius,no_of_valleys,valley_radius,no_of_houses,house_width, house_height,building_labels,building_width, building_height,vertex_weights,rectilinear)


    # Brute Force TSP
    if no_of_buildings <= compute_capacity:
        optimal_tour, optimal_length = brute_force_tsp(G)
        brute_force_tsp_image = draw_tsp_route(image.copy(), optimal_tour, center_points, rectilinear)
    else:
        optimal_tour, optimal_length = None, None
        brute_force_tsp_image = None

    # Nearest Neighbor TSP
    nnt_tour, nnt_length = nearest_neighbor_tsp_algorithm(G)
    nn_tsp_image = draw_tsp_route(image.copy(), nnt_tour, center_points, rectilinear)

    # 2-opt TSP
    two_opt_tour, two_opt_length = two_opt_algorithm(G)
    two_opt_tsp_image = draw_tsp_route(image.copy(), two_opt_tour, center_points, rectilinear)

    # MST 
    mst_tour, mst_length = kruskal_with_vertex_weights(G)
    mst_image =  draw_mst_route(image.copy(), mst_tour, center_points, rectilinear)

    return optimal_tour, optimal_length, nnt_tour, nnt_length, two_opt_tour, two_opt_length, mst_tour, mst_length, image_with_graph_weights, image, brute_force_tsp_image, nn_tsp_image, two_opt_tsp_image, mst_image


def main():

    st.set_page_config(page_title="Navigating the Urban Jungle", page_icon=":car:")

    acceptable_labels = ["Hospital", "Gas Station", "Public Park", "School"]

    st.title("Navigating the Urban Jungle : Optimizing Road Network")

    st.sidebar.header("Input Parameters")
    no_of_houses = st.sidebar.slider("Number of houses", 2, 50, 10)
    no_of_valleys = st.sidebar.slider("Number of valleys", 0, 10, 1)
    no_of_hills = st.sidebar.slider("Number of hills", 0, 10, 1)

    st.sidebar.header("Select Public Amenities")
    toggle = st.sidebar.checkbox("Use Default Public Amenities", value=True)

    if toggle:
        building_labels = ["Hospital", "Gas Station", "Public Park", "School"]
        vertex_weights = {"House": 1, "Hospital": 2, "Gas Station": 3, "Public Park": 4, "School": 5}

    else:
        selected_items = st.sidebar.multiselect("Choose items", acceptable_labels)

        bulildings = {}
        vertex_weights = {"House": 1}
        for item_name in selected_items:
            col1,col2= st.sidebar.columns(2)
            amount = col1.number_input(f"Enter Number of {item_name}", min_value=0, value=1)
            weight_ = col2.number_input(f"Enter weight of {item_name}", min_value=0,value=1,help = "The Frequency of visits to this place")
            bulildings[item_name] = amount
            vertex_weights[item_name] = weight_

        building_labels = []
        for key, value in bulildings.items():
            building_labels.extend([key] * value)


    st.sidebar.header("Distance Metric")
    toggle = st.sidebar.checkbox("Use Rectilinear Distance", value=True)
    if toggle:
        grid_cell_size = st.sidebar.slider("Grid cell size", 10, 50, 30, 10)
        rectilinear = True
    else:
        grid_cell_size = None
        rectilinear = False

    st.sidebar.header("Size Parameters")
    col3,col4 = st.sidebar.columns(2)
    house_width = col3.number_input("House width", value=20)
    house_height = col4.number_input("House height", value=20)
    hill_radius = col3.number_input("Hill radius", value=30)
    valley_radius = col4.number_input("Valley radius", value=20)

    building_width, building_height = random.randint(30, 60), random.randint(30, 60)
    compute_capacity = 10
    no_of_buildings = len(building_labels) + no_of_houses

    vars_ = [rectilinear,house_width,house_height,hill_radius,valley_radius,building_width,building_height,compute_capacity,building_labels,vertex_weights,no_of_houses,no_of_valleys,no_of_hills]

    if all(item is not None for item in vars_):
        optimal_tour, optimal_length, nnt_tour, nnt_length, two_opt_tour, two_opt_length, mst_tour, mst_length, image_with_graph_weights, image, brute_force_tsp_image, nn_tsp_image, two_opt_tsp_image, mst_image = generate_optimal_road_network(rectilinear,grid_cell_size,house_width,house_height,hill_radius,valley_radius,building_width,building_height,compute_capacity,building_labels,vertex_weights,no_of_houses,no_of_valleys,no_of_hills,no_of_buildings)

        st.markdown("<h3 style='text-align: center;'>Fully Connected Graph</h3>", unsafe_allow_html=True)
        image_with_graph_weights = Image.fromarray(image_with_graph_weights)
        st.image(image_with_graph_weights, caption="Connected graph with vertex and edge weights", use_column_width=True)


        st.markdown("<h3 style='text-align: center;'>Optimal Solutions</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        mst_image = Image.fromarray(mst_image)
        col1.image(mst_image, caption= f"Minimum Spanning Tree", use_column_width=True)
        col1.markdown(f"<h5 style='text-align: center;'>weight : {round(mst_length,2)}</h5>", unsafe_allow_html=True)

        if no_of_buildings <= compute_capacity:
            brute_force_tsp_image = Image.fromarray(brute_force_tsp_image)
            col2.image(brute_force_tsp_image, caption=f"Brute Force TSP Solution", use_column_width=True)
            col2.markdown(f"<h5 style='text-align: center;'>weight : {round(optimal_length,2)}</h5>", unsafe_allow_html=True)
        else:
            with col2:
                st.warning(f"Due to Resource Constraint, Brute Force TSP Solution can not computed for more than {compute_capacity} houses")
                

        st.markdown("<h3 style='text-align: center;'>Approximate Solutions</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        two_opt_tsp_image = Image.fromarray(two_opt_tsp_image)
        col1.image(two_opt_tsp_image, caption=f"2-Opt TSP Solution", use_column_width=True)
        col1.markdown(f"<h5 style='text-align: center;'>weight : {round(two_opt_length,2)}</h5>", unsafe_allow_html=True)

        nn_tsp_image = Image.fromarray(nn_tsp_image)
        col2.image(nn_tsp_image, caption=f"NN TSP Solution", use_column_width=True)
        col2.markdown(f"<h5 style='text-align: center;'>weight : {round(nnt_length,2)}</h5>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()