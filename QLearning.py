import numpy as np
import random
import json
import time
import math
import os
from IOReader import Data

class QLearningSolver:
    def __init__(self, data_model: Data, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.data = data_model
        self.data.calcDistMat()
        
        # Q-Learning Parameters
        self.num_nodes = self.data.n
        self.alpha = alpha          
        self.gamma = gamma          
        self.epsilon = epsilon      
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-Table [Current Node][Next Node]
        self.q_table = np.zeros((self.num_nodes, self.num_nodes))
        
        # Environment Settings
        self.speed = 1.0
        self.DAY_LENGTH = 12.0 # Match OR-Tools day length
        self.max_fleet_capacity = max([ship['Capacity'] for ship in self.data.ships])

    def get_valid_actions(self, current_node, unvisited, current_load, current_time):
        valid_nodes = []
        for node in unvisited:
            # 1. Check Capacity
            demand = self.data.demands[node]
            if current_load + demand > self.max_fleet_capacity:
                continue
            
            # 2. Check Time Window
            dist = self.data.dist_matrix[current_node][node]
            travel_time = dist / self.speed
            arrival_time = max(current_time + travel_time, self.data.time_windows[node][0])
            
            if arrival_time <= self.data.time_windows[node][1]:
                valid_nodes.append(node)
        return valid_nodes

    def choose_action(self, current_node, valid_nodes):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_nodes)
        
        q_values = {node: self.q_table[current_node][node] for node in valid_nodes}
        max_q = max(q_values.values())
        best_candidates = [node for node, q in q_values.items() if q == max_q]
        return random.choice(best_candidates)

    def learn(self, state, action, reward, next_state, possible_future_nodes):
        old_value = self.q_table[state][action]
        if possible_future_nodes:
            future_max = max([self.q_table[next_state][n] for n in possible_future_nodes])
        else:
            future_max = 0 
        new_value = old_value + self.alpha * (reward + self.gamma * future_max - old_value)
        self.q_table[state][action] = new_value

    def assign_vehicle(self, route_load):
        best_vehicle = None
        min_rent = float('inf')
        for ship in self.data.ships:
            if ship['Capacity'] >= route_load:
                if ship['Rent'] < min_rent:
                    min_rent = ship['Rent']
                    best_vehicle = ship
        if best_vehicle is None:
            best_vehicle = self.data.ships[-1] 
        return best_vehicle

    def calculate_solution_metrics(self, routes):
        """
        Takes a list of routes (lists of node indices) and returns detailed metrics
        including formatted path lists, specific vehicle assignments, and total cost.
        """
        total_cost = 0
        total_dist = 0
        detailed_routes = []

        for route_indices in routes:
            # 1. Calculate Load
            route_load = sum(self.data.demands[n] for n in route_indices)
            
            # 2. Assign Vehicle
            vehicle = self.assign_vehicle(route_load)
            base_rent = vehicle['Rent']

            # 3. Calculate Time & Duration for Day Rate
            curr_node = 0
            curr_time = self.data.time_windows[0][0] # Start time
            route_d = 0
            path_ids = [self.data.ids[0]] # Start at Depot ID

            # Start time of the route (Depot departure)
            start_time = curr_time 

            for node in route_indices:
                d = self.data.dist_matrix[curr_node][node]
                route_d += d
                
                travel_time = d / self.speed
                arrival_time = max(curr_time + travel_time, self.data.time_windows[node][0])
                curr_time = arrival_time + self.data.service_times[node]
                
                curr_node = node
                path_ids.append(self.data.ids[node])

            # Return to Depot
            d_end = self.data.dist_matrix[curr_node][0]
            route_d += d_end
            curr_time += (d_end / self.speed)
            path_ids.append(self.data.ids[0])

            end_time = curr_time
            duration = end_time - start_time
            
            # 4. Apply Day Rate Logic
            days_billed = math.ceil(duration / self.DAY_LENGTH)
            if days_billed == 0: days_billed = 1
            
            final_rent = base_rent * days_billed
            
            total_cost += final_rent
            total_dist += route_d

            detailed_routes.append({
                "type": vehicle['Type'],
                "path": path_ids,
                "load": route_load,
                "cost": final_rent,
                "days_billed": days_billed,
                "distance": route_d
            })

        return detailed_routes, total_cost

    def solve(self, csv_path, episodes=2000):
        print(f"Training Q-Learning Agent over {episodes} episodes...")
        
        # Metrics Storage
        initial_solution_data = None
        initial_cost = 0
        
        best_global_cost = float('inf')
        best_solution_data = None

        # --- START TIMER ---
        start_time = time.time()

        for ep in range(episodes):
            unvisited = set(self.data.ids)
            unvisited.remove(0) 
            
            routes = []
            
            while unvisited:
                current_node = 0 
                current_load = 0
                current_time = self.data.time_windows[0][0]
                route = [] 
                
                while True:
                    valid_nodes = self.get_valid_actions(current_node, unvisited, current_load, current_time)
                    
                    if not valid_nodes:
                        next_node = 0
                    else:
                        next_node = self.choose_action(current_node, valid_nodes)
                    
                    dist = self.data.dist_matrix[current_node][next_node]
                    reward = -dist 
                    
                    if next_node != 0:
                        travel_time = dist / self.speed
                        arrival_time = max(current_time + travel_time, self.data.time_windows[next_node][0])
                        service_time = self.data.service_times[next_node]
                        
                        future_unvisited = unvisited.copy()
                        future_unvisited.remove(next_node)
                        future_valid = self.get_valid_actions(next_node, future_unvisited, 
                                                              current_load + self.data.demands[next_node], 
                                                              arrival_time + service_time)
                        
                        self.learn(current_node, next_node, reward, next_node, future_valid)
                        
                        current_node = next_node
                        current_load += self.data.demands[next_node]
                        current_time = arrival_time + service_time
                        unvisited.remove(next_node)
                        route.append(current_node)
                    else:
                        self.learn(current_node, 0, reward, 0, [])
                        break 
                
                routes.append(route)
            
            # --- METRIC CALCULATION ---
            # Calculate precise cost for this episode's solution
            detailed_routes, episode_cost = self.calculate_solution_metrics(routes)

            # Capture Episode 0
            if ep == 0:
                initial_solution_data = detailed_routes
                initial_cost = episode_cost

            # Update Global Best (Minimizing COST, not just distance)
            if episode_cost < best_global_cost:
                best_global_cost = episode_cost
                best_solution_data = detailed_routes
            
            # Decay Epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # --- END TIMER ---
        executing_time = time.time() - start_time
        
        # --- JSON GENERATION ---
        self.save_results(
            csv_path, 
            initial_solution_data, 
            initial_cost, 
            best_solution_data, 
            best_global_cost, 
            executing_time
        )

    def save_results(self, data_path, init_sol, init_cost, final_sol, final_cost, exec_time):
        RESULT_DIR = "result/"
        # 1. Read OR-Tools result for comparison
        ortools_cost = 0
        cost_diff = 0
        ortools_file = RESULT_DIR + "ortools_20.json"
        
        if os.path.exists(ortools_file):
            try:
                with open(ortools_file, 'r') as f:
                    or_data = json.load(f)
                    ortools_cost = or_data.get("final_global_optimal_cost", 0)
                    # Diff = QLearning Cost - ORTools Cost
                    cost_diff = final_cost - ortools_cost
            except Exception:
                cost_diff = "Error reading ORTools file"
        else:
            cost_diff = "ORTOOLS file not found"

        # 2. Construct JSON
        output_data = {
            "algorithm": "QLearning",
            "data_path": data_path,
            "initial_global_optimal_solution": init_sol,
            "final_global_optimal_solution": final_sol,
            "initial_global_optimal_cost": init_cost,
            "final_global_optimal_cost": final_cost,
            "cost_diff": cost_diff,
            "executing_time": exec_time
        }

        # 3. Write JSON
        output_path = RESULT_DIR + "qlearning_20.json"
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        # 4. Console Summary
        print("\n" + "="*60)
        print(f"{'FINAL RESULTS':^60}")
        print("="*60)
        print(f"Initial Cost (Ep 0) : ${init_cost:,.0f}")
        print(f"Final Best Cost     : ${final_cost:,.0f}")
        print(f"Execution Time      : {exec_time:.4f} seconds")
        print(f"Results saved to    : {output_path}")
        print("="*60)


if __name__ == "__main__":
    csv_path = "datasets/marine_debris_20.csv"
    
    try:
        data_obj = Data(csv_path)
        # episodes=3000 allows decent convergence
        agent = QLearningSolver(data_obj, alpha=0.1, gamma=0.9) 
        agent.solve(csv_path, episodes=3000)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")