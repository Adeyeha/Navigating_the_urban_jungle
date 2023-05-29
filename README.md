# Project Name: Navigating the Urban Jungle - Optimizing Road Networks

## Description
This project implements a graph-based routing algorithm for urban environments. It aims to find the optimal route through a city, considering various obstacles and landmarks. The algorithm uses a graph representation of the city's topology, where nodes represent key locations and edges represent the connections between them. The algorithm takes into account factors such as distances, weights, and obstacles to determine the most efficient path.

The project includes the following main components:

1. Graph Creation: The city topology is represented as a graph, where nodes represent locations (houses, buildings) and edges represent connections between them. The graph is created using the NetworkX library and is initialized with randomly generated buildings, houses, and obstacles (hills and valleys).
2. Vertex and Edge Weights: Each node in the graph is assigned a weight based on its type (e.g., house, building) and importance (e.g., hospital, school). The edges are assigned weights based on distance and obstacles. These weights are used to calculate the optimal route.
3. Routing Algorithms: The project implements several routing algorithms, including brute-force TSP, nearest-neighbor TSP, and 2-opt. These algorithms aim to find the most efficient path through the city by considering different criteria, such as minimizing distance or time.
4. Visualization: The project includes visualization components using the OpenCV library to display the city layout, the graph, and the calculated routes.

## Problem Formulation (Integer Linear Programming)
To find the optimal route through the city, an Integer Linear Programming (ILP) formulation can be used. Here is the problem formulation:

- ### Input:
    A graph G = (V, E) representing the city topology, where V is the set of nodes and E is the set of edges.
    Each node v ∈ V represents a location (house, building) in the city.
    Each edge (u, v) ∈ E represents a connection between nodes u and v.
    Each node v ∈ V is assigned a weight w(v) based on its type and importance.
    Each edge (u, v) ∈ E is assigned a weight w(u, v) based on distance and obstacles.

- ### Variables:
    x(u, v) ∈ {0, 1}: Binary variable indicating whether edge (u, v) is included in the route.

- ### Objective:
    Minimize the total weight of the route:
    Minimize: ∑(u,v) ∈ E w(u, v) * x(u, v)

- ### Constraints:
    1. Each node v ∈ V should be visited exactly once:
        - ∑(u,v) ∈ E x(u, v) = 1   for all v ∈ V

    2. Each node u ∈ V should be visited exactly once:
        - ∑(u,v) ∈ E x(u, v) = 1   for all u ∈ V

    3. Subtour elimination constraint to prevent cycles:
        - ∑(u,v) ∈ S x(u, v) ≤ |S| - 1   for all S ⊆ V, 2 ≤ |S| ≤ |V| - 1

    4. Binary constraints on the variables:
        - x(u, v) ∈ {0, 1}   for all (u, v) ∈ E

Solving this ILP formulation will provide the optimal route through the city, considering the weights assigned to nodes and edges. The objective function aims to minimize the total weight, and the constraints ensure that each node is visited exactly once and prevent the formation of subtours.

## How It Works
### Graph Creation:
- The city layout is randomly generated, including houses, buildings, hills, and valleys.
- Randomly positioned circular obstacles (hills and valleys) are generated.
- Houses and buildings are placed on the city grid, ensuring they do not intersect with other existing shapes.
- Buildings are labeled with specific types (e.g., hospital, gas station) and assigned weights based on their importance.
- The city topology is represented as a complete graph, where nodes represent the center points of houses, buildings, and obstacles. Edge weights are assigned based on distance and obstacles.

### Routing Algorithms:
- The project provides different routing algorithms to find the optimal route through the city.
  - Kruskal MST: This algorithm constructs a minimum spanning tree (MST) of the graph by iteratively adding edges with the lowest weight that do not form cycles. The resulting MST can be used as a backbone for finding efficient routes through the city.
  - Brute-Force TSP: This algorithm exhaustively searches all possible permutations of the nodes to find the tour with the minimum length.
  - Nearest Neighbor TSP: This algorithm starts from a randomly chosen node and iteratively selects the nearest unvisited neighbor until all nodes are visited, forming a tour.
  - 2-Opt TSP: This algorithm iteratively improves an initial tour by swapping pairs of edges to reduce the total tour length.

### Visualization:
- The city layout and graph are visualized using the OpenCV library.
- The graph is displayed with edges representing the connections between nodes.
- Vertex and edge weights are displayed as labels on the graph.
- The calculated routes are visualized on the city layout, highlighting the optimal path.

## Installation
To run the project, follow these steps:

1. Clone the repository: `git clone <repository_url>`
2. Install the required dependencies:
   - OpenCV: `pip install opencv-python`
   - NumPy: `pip install numpy`
   - Shapely: `pip install shapely`
   - NetworkX: `pip install networkx`
3. Run the main script: `python models/tsp_v2.py`

## Future Improvements

The current implementation provides a basic framework for graph-based routing in urban environments. Here are some potential areas for future improvement and expansion:

1. Improved Topography Modeling: Enhance the simulator to model more realistic topographies and terrains, taking into account factors such as elevation, slope, and land features. This will enable the algorithm to better handle urban environments with diverse landscapes.

2. Algorithm Expansion: Expand the range of algorithms available in the toolkit. Explore advanced routing algorithms that consider additional factors such as traffic congestion, road conditions, user preferences, and real-time data. This will provide more options for finding optimal routes based on specific requirements and constraints.

3. Long-Range Road Networks: Adapt the simulator to handle long-range road networks, including intercity or interstate routes. This will involve incorporating large-scale road maps, highway systems, and complex urban structures into the algorithm. By considering long-range connectivity, the routing algorithm can address the challenges and requirements of intercity or regional travel.

4. Customization and Specific Requirements: Develop features to allow customization of the routing algorithm for specific requirements, such as emergency response routing, logistics planning, or transportation optimization. This will enable users to adapt the algorithm to their specific use cases and tailor it to their unique needs.

These focus areas will further enhance the capabilities of the graph-based routing algorithm, making it more versatile and applicable to a wider range of urban environments and transportation scenarios.


## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/new-feature`.
3. Make your changes and commit them: `git commit -am 'Add new feature'`.
4. Push to the branch: `git push origin feature/new-feature`.
5. Submit a pull request.

## Contact
If you have any questions or suggestions regarding this project, please feel free to contact us:

- Temitope: [mantupee@gmail.com]
- Uthman: [Uthmanjinadu@gmail.com]
