# Single-function OR-Tools CVRPTW solver matched to your CSV data layout
import pandas as pd
import math
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def solve_vrptw_ortools(
    filepath: str,
    depot_coord: tuple = (10.762622, 106.660172),   # (lat, lon) ; set to None if depot already in CSV
    include_depot: bool = True,
    vehicle_capacities: list = None,  # e.g. [20, 10, 5] (tons)
    speed_kmph: float = 20.0,         # travel speed in km/h used to convert distance->time
    time_limit_seconds: int = 30      # solver time limit
):
    """
    Read CSV and solve CVRPTW with OR-Tools. Returns (routes, total_distance_km).
    CSV expected columns (exact names): 
      'no.', 'latitude(N)', 'longitude(E)', 'weight(tons)', 'collection_time(h)', 'time_windows'
    time_windows entries like "8.0-12.0" (hours). Service times in hours.
    If include_depot=True, the depot_coord is inserted as node 0 (demand=0, tw=(0,24), service=0).
    vehicle_capacities: list of integers (tons). Number of vehicles = len(vehicle_capacities).
    """
    # ------------------------
    # 1. Read CSV
    # ------------------------
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    # Keep original IDs if present
    if 'no.' in df.columns:
        ids = df['no.'].tolist()
    else:
        ids = list(range(1, len(df) + 1))

    coords = list(zip(df['latitude(N)'].astype(float), df['longitude(E)'].astype(float)))
    demands = df['weight(tons)'].astype(float).tolist() if 'weight(tons)' in df.columns else [0.0]*len(coords)
    service_times_h = df['collection_time(h)'].astype(float).tolist() if 'collection_time(h)' in df.columns else [0.0]*len(coords)

    # Parse time windows; fallback to full day if missing/invalid
    time_windows_h = []
    for tw in df.get('time_windows', [None]*len(df)):
        try:
            start, end = map(float, str(tw).split('-'))
            time_windows_h.append((start, end))
        except Exception:
            time_windows_h.append((0.0, 24.0))

    # ------------------------
    # 2. Optionally prepend depot
    # ------------------------
    node_ids = []
    node_coords = []
    node_demands = []
    node_service_h = []
    node_tw_h = []

    if include_depot:
        node_ids.append(0)  # depot id
        node_coords.append(depot_coord)
        node_demands.append(0.0)
        node_service_h.append(0.0)
        node_tw_h.append((0.0, 24.0))

    # then customers
    for i in range(len(coords)):
        node_ids.append(ids[i])
        node_coords.append(coords[i])
        node_demands.append(demands[i])
        node_service_h.append(service_times_h[i])
        node_tw_h.append(time_windows_h[i])

    num_nodes = len(node_coords)

    # ------------------------
    # 3. Distances (Haversine -> km) and travel times (minutes)
    # ------------------------
    def haversine_km(a, b):
        lat1, lon1 = a
        lat2, lon2 = b
        R = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        x = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
        return 2*R*math.asin(math.sqrt(x))

    # Build distance matrix in km and travel time matrix in minutes (integers)
    dist_km = [[0.0]*num_nodes for _ in range(num_nodes)]
    travel_time_min = [[0]*num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            d = haversine_km(node_coords[i], node_coords[j]) if i != j else 0.0
            dist_km[i][j] = d
            # travel time in minutes = (km / kmph) * 60
            tmin = 0 if i == j else int(round((d / max(1e-6, speed_kmph)) * 60.0))
            travel_time_min[i][j] = tmin

    # ------------------------
    # 4. Prepare OR-Tools data model
    # ------------------------
    if vehicle_capacities is None:
        # default: one vehicle with capacity equal to sum(demands)
        total_demand = int(math.ceil(sum(node_demands)))
        vehicle_capacities = [total_demand]

    num_vehicles = len(vehicle_capacities)
    depot_index = 0  # by construction

    # demands must be ints for OR-Tools
    demands_int = [int(round(d)) for d in node_demands]
    capacities_int = [int(round(c)) for c in vehicle_capacities]
    service_time_min = [int(round(h*60.0)) for h in node_service_h]
    tw_min = [(int(round(a*60.0)), int(round(b*60.0))) for (a,b) in node_tw_h]  # in minutes

    # ------------------------
    # 5. Create routing model
    # ------------------------
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback (for objective)
    def distance_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        # Convert km to integer (we can keep float objective by using travel_time instead,
        # but we'll keep km*1000 as integer)
        return int(round(dist_km[f][t] * 1000))  # objective units: meters approx

    dist_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback_index)

    # Capacity dimension (demands)
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands_int[from_node]
    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx, 0, capacities_int, True, 'Capacity'
    )

    # Time dimension (travel time + service time) with time windows
    def time_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return travel_time_min[f][t] + service_time_min[f]
    time_cb_idx = routing.RegisterTransitCallback(time_callback)
    # allow global slack (max waiting) as large number
    horizon = 24*60  # 1 day in minutes
    routing.AddDimension(
        time_cb_idx,
        int(horizon),  # allow waiting slack up to horizon
        int(horizon),  # maximum time per vehicle
        False,         # Don't force start cumul to zero (we'll set windows)
        'Time'
    )
    time_dim = routing.GetDimensionOrDie('Time')

    # Add time window constraints for each node
    for node_idx in range(num_nodes):
        index = manager.NodeToIndex(node_idx)
        start, end = tw_min[node_idx]
        time_dim.CumulVar(index).SetRange(start, end)

    # Allow vehicles to start at depot at time 0..horizon
    for v in range(num_vehicles):
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(v)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    # Setting first solution and search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(time_limit_seconds)
    search_parameters.log_search = False

    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        raise RuntimeError("No solution found by OR-Tools within time limit.")

    # ------------------------
    # 6. Extract routes & compute total distance
    # ------------------------
    routes = []
    total_distance = 0.0  # in km

    for v in range(num_vehicles):
        index = routing.Start(v)
        route_nodes = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_nodes.append(node_ids[node])  # append original ID (0 for depot or CSV id)
            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)
            total_distance += dist_km[node][next_node]
            index = next_index
        # append last depot (end)
        route_nodes.append(node_ids[manager.IndexToNode(index)])
        # Only keep meaningful routes (more than depot->depot)
        if len(route_nodes) > 2 or (len(route_nodes) == 2 and route_nodes[0] != route_nodes[1]):
            routes.append(route_nodes)

    return routes, total_distance

# ------------------------
# Example usage:
# ------------------------
# routes, tot_km = solve_vrptw_ortools(
#     "datasets/marine_debris_20.csv",
#     depot_coord=(10.762622, 106.660172),
#     include_depot=True,
#     vehicle_capacities=[20, 10, 5],
#     speed_kmph=15.0,
#     time_limit_seconds=20
# )
# print("Routes:", routes)
# print("Total distance (km):", round(tot_km,2))
