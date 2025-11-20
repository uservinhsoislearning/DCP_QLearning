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
        
        # --- NEW COST CONSTANTS ---
        # You can adjust these values as needed
        self.FUEL_COST_PER_KM_TON = 0.5  # $0.50 per km per ton of weight
        self.LABOR_COST_PER_HOUR = 20.0  # $20.00 per hour for the worker
        
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
        # --- START TIMER ---
        start_time = time.time()

        manager = pywrapcp.RoutingIndexManager(
            self.data.n, 
            self.num_vehicles, 
            self.depot_index
        )
        routing = pywrapcp.RoutingModel(manager)

        # --- 1. DEFINE COSTS IN DOLLARS (SCALED) ---
        # We scale everything to keep units consistent (e.g., all in "Milli-Dollars")
        
        # ESTIMATE: Average load factor to approximate fuel cost during search
        # Since we can't easily do dynamic load cost in search, we assume ships are half-full on average
        AVG_LOAD_ESTIMATE = sum(self.data.demands) / self.num_vehicles / 2
        
        # Cost per Unit of Distance (Fuel)
        # Cost = Dist * Fuel_Rate * Avg_Load
        fuel_cost_per_unit_dist = self.FUEL_COST_PER_KM_TON * AVG_LOAD_ESTIMATE

        # --- 2. CALLBACKS ---

        # A. ARC COST CALLBACK (Represents Fuel Cost $$)
        def fuel_cost_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            dist = self.data.dist_matrix[from_node][to_node]
            
            # Calculate approximate fuel cost for this leg
            cost_dollars = dist * fuel_cost_per_unit_dist
            
            # Return as integer scaled value
            return int(cost_dollars * self.SCALING_FACTOR)

        transit_callback_index = routing.RegisterTransitCallback(fuel_cost_callback)
        
        # Set this as the primary objective to minimize
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


        # B. DEMAND CALLBACK (Standard)
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(self.data.demands[from_node] * self.SCALING_FACTOR)
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)


        # C. TIME CALLBACK (Standard calculation, needed for constraints)
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            dist_val = self.data.dist_matrix[from_node][to_node]
            travel_time = dist_val / self.VEHICLE_SPEED
            service_time = self.data.service_times[from_node]
            return int((travel_time + service_time) * self.SCALING_FACTOR)
        time_callback_index = routing.RegisterTransitCallback(time_callback)

        # --- 3. DIMENSIONS ---

        # Capacity
        vehicle_capacities = [int(v["Capacity"] * self.SCALING_FACTOR) for v in self.vehicles]
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, vehicle_capacities, True, "Capacity"
        )

        # Time Dimension (Add LABOR COST here)
        horizon_int = int(100000.0 * self.SCALING_FACTOR)
        routing.AddDimension(
            time_callback_index, horizon_int, horizon_int, False, "Time"
        )
        time_dimension = routing.GetDimensionOrDie("Time")

        # *** CRITICAL FIX: LABOR COST ***
        # We tell the solver: "Every 1 unit of Time adds $20 to the Objective"
        # Since time is already scaled by SCALING_FACTOR, we just pass the raw dollar rate
        # OR-Tools multiplies this coefficient by the CumulVar (Time).
        # Note: We usually apply this to the Span (Total Time), but strict hourly is tricky.
        # GlobalSpanCostCoefficient penalizes the *Total Fleet Duration*.
        
        # Set coefficient to Labor Cost per Hour
        time_dimension.SetGlobalSpanCostCoefficient(int(self.LABOR_COST_PER_HOUR))

        # Time Windows
        for location_idx, (start, end) in enumerate(self.data.time_windows):
            if location_idx == 0: continue 
            end_val = horizon_int if end == float('inf') else int(end * self.SCALING_FACTOR)
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(int(start * self.SCALING_FACTOR), end_val)

        # Depot Time Window
        depot_idx = manager.NodeToIndex(self.data.ids.index(0))
        time_dimension.CumulVar(depot_idx).SetRange(
            int(self.data.time_windows[0][0] * self.SCALING_FACTOR),
            horizon_int
        )

        # --- 4. FIXED COSTS (RENT) ---
        # The solver adds this value to the Objective if the vehicle is used.
        # Since our Arc Costs and Time Costs are now roughly in "Dollars * ScalingFactor",
        # We must ensure Rent is also "Dollars * ScalingFactor"
        for vehicle_id, vehicle_data in enumerate(self.vehicles):
            cost_scaled = int(vehicle_data["Rent"] * self.SCALING_FACTOR)
            routing.SetFixedCostOfVehicle(cost_scaled, vehicle_id)

        # --- 5. SOLVE ---
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = 30

        solution = routing.SolveWithParameters(search_parameters)
        
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
        Extracts data and calculates detailed costs:
        1. Rent (Fixed Daily Rate)
        2. Labor (Hourly Rate * Duration)
        3. Fuel (Distance * Current Load * Fuel Rate)
        """
        total_combined_cost = 0
        total_rent_cost = 0
        total_labor_cost = 0
        total_fuel_cost = 0
        
        time_dimension = routing.GetDimensionOrDie("Time")
        
        routes_data = []

        for vehicle_id in range(self.num_vehicles):
            index = routing.Start(vehicle_id)
            
            if routing.IsEnd(solution.Value(routing.NextVar(index))):
                continue

            vehicle_info = self.vehicles[vehicle_id]
            base_rent = vehicle_info['Rent']
            
            route_path = []
            route_distance = 0
            route_fuel_cost = 0
            
            # "running_load" tracks the weight currently on the ship
            running_load = 0 
            
            # Start Time
            start_time_var = time_dimension.CumulVar(index)
            start_time = solution.Min(start_time_var) / self.SCALING_FACTOR
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_path.append(self.data.ids[node_index])
                
                prev_index = index
                index = solution.Value(routing.NextVar(index))
                
                # Calculate Leg Data
                from_node = manager.IndexToNode(prev_index)
                to_node = manager.IndexToNode(index)
                leg_dist = self.data.dist_matrix[from_node][to_node]
                
                # --- FUEL CALCULATION ---
                # Cost to move the CURRENT load across this distance
                # (Assumes we carry the load picked up so far)
                route_fuel_cost += (leg_dist * running_load * self.FUEL_COST_PER_KM_TON)
                
                # Add new load collected at the destination node (Pickup)
                running_load += self.data.demands[to_node]
                
                route_distance += leg_dist

            # Add Return to Depot
            node_index = manager.IndexToNode(index)
            route_path.append(self.data.ids[node_index])
            
            # Note: Return trip to depot carries the MAX load
            # (Depot is node 0, so demands[0] is 0, load doesn't increase)
            
            # End Time
            end_time_var = time_dimension.CumulVar(index)
            end_time = solution.Min(end_time_var) / self.SCALING_FACTOR
            
            # --- TIME & LABOR CALCULATION ---
            duration = end_time - start_time
            labor_cost = duration * self.LABOR_COST_PER_HOUR
            
            # --- RENT CALCULATION ---
            days_billed = math.ceil(duration / self.DAY_LENGTH)
            if days_billed == 0: days_billed = 1
            rent_cost = base_rent * days_billed
            
            # --- TOTALS ---
            vehicle_total_cost = rent_cost + labor_cost + route_fuel_cost
            
            total_rent_cost += rent_cost
            total_labor_cost += labor_cost
            total_fuel_cost += route_fuel_cost
            total_combined_cost += vehicle_total_cost
            
            routes_data.append({
                "type": vehicle_info['Type'],
                "path": route_path,
                "final_load": running_load,
                "distance": route_distance,
                "duration_hours": duration,
                "days_billed": days_billed,
                "cost": total_combined_cost
            })

        self.print_console_summary(total_combined_cost, total_rent_cost, total_labor_cost, total_fuel_cost, routes_data)

        return {
            "routes": routes_data,
            "total_cost": total_combined_cost
        }

    def print_console_summary(self, total, rent, labor, fuel, routes):
        print(f"\n{' SOLUTION SUMMARY ':=^60}")
        for r in routes:
            print(f"Ship ({r['type']}): Path {r['path']}")
            print(f"   > Load: {r['final_load']:.2f}t | Dist: {r['distance']:.2f}km | Time: {r['duration_hours']:.2f}h")
            print(f"   > Total_cost: {r['cost']:.2f}")
            print("-" * 30)
            
        print("=" * 60)

if __name__ == "__main__":
    # 1. Config
    csv_path = "/kaggle/input/dataset/marine_debris_100.csv"
    json_output_path = "ortools_100.json"

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
                "cost_diff": 0.0,
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