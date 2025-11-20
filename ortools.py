import sys
import math
import time
import json
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import IOReader as io

class CVRPTWSolver:
    def __init__(self, data_model: io.Data):
        self.data = data_model
        self.data.calcDistMat() 
        
        # --- CONFIGURATION ---
        self.SCALING_FACTOR = 1000
        self.VEHICLE_SPEED = 1.0
        self.DAY_LENGTH = 12.0  # 12 hours per day billing cycle
        
        # --- FLEET GENERATION ---
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
        """
        Executes the solver and returns a dictionary with solution metrics.
        """
        # --- START TIMER ---
        start_time = time.time()

        # 1. Setup
        manager = pywrapcp.RoutingIndexManager(
            self.data.n, 
            self.num_vehicles, 
            self.depot_index
        )
        routing = pywrapcp.RoutingModel(manager)

        # 2. Callbacks
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(self.data.dist_matrix[from_node][to_node] * self.SCALING_FACTOR)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(self.data.demands[from_node] * self.SCALING_FACTOR)

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            dist_val = self.data.dist_matrix[from_node][to_node]
            travel_time = dist_val / self.VEHICLE_SPEED
            service_time = self.data.service_times[from_node]
            return int((travel_time + service_time) * self.SCALING_FACTOR)

        time_callback_index = routing.RegisterTransitCallback(time_callback)

        # 3. Dimensions
        vehicle_capacities = [int(v["Capacity"] * self.SCALING_FACTOR) for v in self.vehicles]
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, vehicle_capacities, True, "Capacity"
        )

        horizon_int = int(100000.0 * self.SCALING_FACTOR) 
        routing.AddDimension(
            time_callback_index, horizon_int, horizon_int, False, "Time"
        )
        time_dimension = routing.GetDimensionOrDie("Time")
        time_dimension.SetGlobalSpanCostCoefficient(10) # Optional optimization hint

        # Time Windows
        for location_idx, (start, end) in enumerate(self.data.time_windows):
            if location_idx == 0: continue 
            end_val = horizon_int if end == float('inf') else int(end * self.SCALING_FACTOR)
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(int(start * self.SCALING_FACTOR), end_val)

        depot_idx = manager.NodeToIndex(self.data.ids.index(0))
        time_dimension.CumulVar(depot_idx).SetRange(
            int(self.data.time_windows[0][0] * self.SCALING_FACTOR),
            horizon_int
        )

        # Fixed Costs
        for vehicle_id, vehicle_data in enumerate(self.vehicles):
            routing.SetFixedCostOfVehicle(int(vehicle_data["Rent"]), vehicle_id)

        # 4. Solve
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = 30

        solution = routing.SolveWithParameters(search_parameters)
        
        # --- END TIMER ---
        executing_time = time.time() - start_time

        if solution:
            result_data = self.process_solution(manager, routing, solution)
            result_data['executing_time'] = executing_time
            return result_data
        else:
            print("No solution found!")
            return None

    def process_solution(self, manager, routing, solution):
        """
        Extracts data from the OR-Tools solution object into a structured dictionary.
        """
        total_hired_cost = 0
        total_distance = 0
        total_load = 0
        time_dimension = routing.GetDimensionOrDie("Time")
        
        routes_data = []

        for vehicle_id in range(self.num_vehicles):
            index = routing.Start(vehicle_id)
            
            if routing.IsEnd(solution.Value(routing.NextVar(index))):
                continue

            vehicle_info = self.vehicles[vehicle_id]
            base_rent = vehicle_info['Rent']
            
            route_path = []
            route_dist = 0
            route_load = 0
            
            # Start Time
            start_time_var = time_dimension.CumulVar(index)
            start_time = solution.Min(start_time_var) / self.SCALING_FACTOR
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += self.data.demands[node_index]
                route_path.append(self.data.ids[node_index]) # Store as list of IDs
                
                prev_index = index
                index = solution.Value(routing.NextVar(index))
                
                from_node = manager.IndexToNode(prev_index)
                to_node = manager.IndexToNode(index)
                route_dist += self.data.dist_matrix[from_node][to_node]

            # Add Return to Depot
            node_index = manager.IndexToNode(index)
            route_path.append(self.data.ids[node_index])
            
            # End Time & Cost Calc
            end_time_var = time_dimension.CumulVar(index)
            end_time = solution.Min(end_time_var) / self.SCALING_FACTOR
            
            duration = end_time - start_time
            days_billed = math.ceil(duration / self.DAY_LENGTH)
            if days_billed == 0: days_billed = 1
            
            final_cost = base_rent * days_billed
            
            total_hired_cost += final_cost
            total_distance += route_dist
            total_load += route_load
            
            routes_data.append({
                "vehicle_id": vehicle_id,
                "type": vehicle_info['Type'],
                "path": route_path,
                "load": route_load,
                "distance": route_dist,
                "cost": final_cost,
                "days_billed": days_billed
            })

        # Print to console as before (optional, but good for verification)
        self.print_console_summary(total_hired_cost, total_distance, total_load, routes_data)

        return {
            "routes": routes_data,
            "total_cost": total_hired_cost
        }

    def print_console_summary(self, cost, dist, load, routes):
        print(f"\n{' SOLUTION SUMMARY ':=^60}")
        for r in routes:
            print(f"Vehicle {r['vehicle_id']} ({r['type']}): Path {r['path']} | Cost: ${r['cost']:,.0f}")
        print("-" * 60)
        print(f"Total Cost: ${cost:,.0f} | Total Dist: {dist:.2f}")

if __name__ == "__main__":
    # 1. Config
    csv_path = "datasets/marine_debris_20.csv"
    json_output_path = "ortools_20.json"

    try:
        # 2. Load Data
        data_loader = io.Data(csv_path)
        
        # 3. Initialize Solver
        solver = CVRPTWSolver(data_loader)
        
        # 4. Run & Capture
        result = solver.solve()
        
        if result:
            # 5. Construct Final JSON Structure
            output_data = {
                "algorithm": "ORTOOLS",
                "data_path": csv_path,
                "initial_global_optimal_solution": result['routes'], # Same as final for OR-Tools
                "final_global_optimal_solution": result['routes'],
                "initial_global_optimal_cost": result['total_cost'], # Same as final for OR-Tools
                "final_global_optimal_cost": result['total_cost'],
                "executing_time": result['executing_time']
            }
            
            # 6. Write to File
            with open(json_output_path, 'w') as f:
                json.dump(output_data, f, indent=4)
                
            print(f"\nResults successfully saved to {json_output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")