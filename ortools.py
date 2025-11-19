import math
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def solve_vrptw_from_data(
    data_obj,
    speed_kmph: float = 20.0,
    time_limit_seconds: int = 30
):
    """
    OR-Tools CVRPTW solver that uses the Data() object from IOReader.py.
    === Data object attributes required ===
        data_obj.num_nodes
        data_obj.nodes[n] : dict with keys:
            'lat', 'lon'
            'demand'
            'service_time'
            'tw_start', 'tw_end'
        data_obj.dist_mat[i][j]   # distances in km
        data_obj.ships = [{"Type":..., "Rent":..., "Capacity":...}, ...]
    """

    num_nodes = data_obj.num_nodes

    # ---------------------------
    # 1. Build distance & time matrices for OR-Tools
    # ---------------------------
    dist_km = data_obj.dist_mat

    # Convert distance to travel time in minutes
    travel_time_min = [
        [
            0 if i == j else int(round((dist_km[i][j] / max(speed_kmph, 1e-6)) * 60))
            for j in range(num_nodes)
        ]
        for i in range(num_nodes)
    ]

    # Service time in minutes
    service_time_min = [
        int(round(data_obj.nodes[i]['service_time'] * 60))
        for i in range(num_nodes)
    ]

    # Time windows in minutes
    tw_min = [
        (
            int(round(data_obj.nodes[i]["tw_start"] * 60)),
            int(round(data_obj.nodes[i]["tw_end"] * 60)),
        )
        for i in range(num_nodes)
    ]

    # ---------------------------
    # 2. Vehicle capacities from ships
    # ---------------------------
    vehicle_capacities = [ship["Capacity"] for ship in data_obj.ships]
    num_vehicles = len(vehicle_capacities)
    depot_index = 0

    # Demands (must be int)
    demands_int = [int(round(data_obj.nodes[i]["demand"])) for i in range(num_nodes)]

    # ---------------------------
    # 3. Build Routing Model
    # ---------------------------
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback (objective: minimize total meters)
    def distance_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return int(round(dist_km[f][t] * 1000))  # convert km→meters

    dist_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback_index)

    # Capacity constraint
    def demand_callback(from_index):
        return demands_int[manager.IndexToNode(from_index)]

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        0,  # no slack
        vehicle_capacities,
        True,
        "Capacity",
    )

    # Time dimension (travel + service)
    def time_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return travel_time_min[f][t] + service_time_min[f]

    time_cb_idx = routing.RegisterTransitCallback(time_callback)

    horizon = 24 * 60  # one day in minutes
    routing.AddDimension(
        time_cb_idx,
        horizon,  # waiting allowed
        horizon,  # vehicle must finish within
        False,
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Time windows
    for node in range(num_nodes):
        idx = manager.NodeToIndex(node)
        start, end = tw_min[node]
        time_dim.CumulVar(idx).SetRange(start, end)

    # Relax start/end times for all vehicles
    for v in range(num_vehicles):
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(v)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    # ---------------------------
    # 4. Search parameters
    # ---------------------------
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(time_limit_seconds)

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        raise RuntimeError("No feasible solution found!")

    # ---------------------------
    # 5. Extract Routes
    # ---------------------------
    routes = []
    total_km = 0.0

    for v in range(num_vehicles):
        index = routing.Start(v)
        route = []

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)

            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)

            total_km += dist_km[node][next_node]
            index = next_index

        route.append(manager.IndexToNode(index))  # depot at end
        if len(route) > 2:
            routes.append(route)

    return routes, total_km

import IOReader as io

if __name__ == "__main__":
    data = io.Data("datasets/marine_debris_20.csv")  # your existing loader

    routes, tot_km = solve_vrptw_from_data(
        data_obj=data,
        speed_kmph=15,
        time_limit_seconds=20
    )

    print("Total distance:", tot_km)
    for i, r in enumerate(routes):
        print(f"Ship {i+1}: {r}")