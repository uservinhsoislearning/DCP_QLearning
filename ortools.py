import sys
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import IOReader as io

class CVRPTWSolver:
    def __init__(self, data_model: io.Data):
        self.data = data_model
        self.data.calcDistMat() 
        
        # --- CONFIGURATION ---
        self.SCALING_FACTOR = 1000  # To convert floats to integers for OR-Tools
        self.VEHICLE_SPEED = 1.0    # Assumption: 1 distance unit = 1 time unit
        
        # --- FLEET GENERATION ---
        # We generate a large enough fleet pool by cycling through your ship types
        # to guarantee feasibility (worst case: 1 ship per customer).
        self.vehicles = []
        num_customers = len(self.data.ids) - 1
        
        while len(self.vehicles) < num_customers:
            for ship in self.data.ships:
                if len(self.vehicles) < num_customers:
                    self.vehicles.append(ship)
                else:
                    break
                    
        self.num_vehicles = len(self.vehicles)
        self.depot_index = 0

    def solve(self):
        # 1. Create Routing Index Manager
        manager = pywrapcp.RoutingIndexManager(
            self.data.n, 
            self.num_vehicles, 
            self.depot_index
        )

        # 2. Create Routing Model
        routing = pywrapcp.RoutingModel(manager)

        # 3. Define Callbacks
        
        # -- Distance Callback --
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(self.data.dist_matrix[from_node][to_node] * self.SCALING_FACTOR)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # -- Demand Callback --
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(self.data.demands[from_node] * self.SCALING_FACTOR)

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

        # -- Time Callback --
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            dist_val = self.data.dist_matrix[from_node][to_node]
            travel_time = dist_val / self.VEHICLE_SPEED
            service_time = self.data.service_times[from_node]
            return int((travel_time + service_time) * self.SCALING_FACTOR)

        time_callback_index = routing.RegisterTransitCallback(time_callback)

        # 4. Add Dimensions
        
        # Capacity Dimension
        vehicle_capacities = [int(v["Capacity"] * self.SCALING_FACTOR) for v in self.vehicles]
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, vehicle_capacities, True, "Capacity"
        )

        # Time Dimension
        horizon_int = int(100000.0 * self.SCALING_FACTOR) # Large horizon
        routing.AddDimension(
            time_callback_index, horizon_int, horizon_int, False, "Time"
        )
        time_dimension = routing.GetDimensionOrDie("Time")

        # Add Time Windows
        for location_idx, (start, end) in enumerate(self.data.time_windows):
            if location_idx == 0: continue # Skip depot here
            end_val = horizon_int if end == float('inf') else int(end * self.SCALING_FACTOR)
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(int(start * self.SCALING_FACTOR), end_val)

        # Depot Time Window
        depot_idx = manager.NodeToIndex(self.data.ids.index(0))
        time_dimension.CumulVar(depot_idx).SetRange(
            int(self.data.time_windows[0][0] * self.SCALING_FACTOR),
            horizon_int
        )

        # 5. Set Fixed Costs (Vehicle Rent)
        for vehicle_id, vehicle_data in enumerate(self.vehicles):
            routing.SetFixedCostOfVehicle(int(vehicle_data["Rent"]), vehicle_id)

        # 6. Solve
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = 30

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            self.print_solution(manager, routing, solution)
        else:
            print("No solution found!")

    def print_solution(self, manager, routing, solution):
        print(f"Objective (Combined Cost): {solution.ObjectiveValue()}")
        
        total_hired_cost = 0
        total_distance = 0
        total_load = 0
        
        active_vehicles = []

        print("\n" + "="*60)
        print(f"{'VEHICLE USAGE & PATHS':^60}")
        print("="*60)

        for vehicle_id in range(self.num_vehicles):
            index = routing.Start(vehicle_id)
            
            # Check if vehicle is used (if NextVar of start is NOT End, it moved)
            if routing.IsEnd(solution.Value(routing.NextVar(index))):
                continue

            # Gather Vehicle Data
            vehicle_info = self.vehicles[vehicle_id]
            rent = vehicle_info['Rent']
            
            # Accumulate Costs
            total_hired_cost += rent
            
            # Path Tracking
            route_nodes = []
            route_distance = 0
            route_load = 0
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += self.data.demands[node_index]
                
                # Add Real Node ID (from CSV) to path list
                route_nodes.append(str(self.data.ids[node_index]))
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                
                # Calculate distance
                from_node = manager.IndexToNode(previous_index)
                to_node = manager.IndexToNode(index)
                route_distance += self.data.dist_matrix[from_node][to_node]

            # Add final return to depot
            node_index = manager.IndexToNode(index)
            route_nodes.append(str(self.data.ids[node_index]))

            total_distance += route_distance
            total_load += route_load

            # Print Route Details
            print(f"\n[Vehicle ID: {vehicle_id}]")
            print(f"   Type     : {vehicle_info['Type']}")
            print(f"   Rent     : ${rent:,.0f}")
            print(f"   Capacity : {vehicle_info['Capacity']} tons")
            print(f"   Load     : {route_load:.2f} tons")
            print(f"   Distance : {route_distance:.2f} km")
            print(f"   Path     : {' -> '.join(route_nodes)}")

        print("\n" + "="*60)
        print(f"{'FINAL SUMMARY':^60}")
        print("="*60)
        print(f"Total Hired Cost (Rent) : ${total_hired_cost:,.0f}")
        print(f"Total Distance Travelled: {total_distance:.2f} km")
        print(f"Total Cargo Delivered   : {total_load:.2f} tons")
        print("="*60)

if __name__ == "__main__":
    # Replace with your actual CSV path
    csv_path = "/kaggle/input/dataset/marine_debris_20.csv" 
    
    try:
        data_loader = io.Data(csv_path)
        solver = CVRPTWSolver(data_loader)
        solver.solve()
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")