import numpy as np
import random
from IOReader import Data

class QLearningSolver:
    def __init__(self, data_model: Data, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.data = data_model
        self.data.calcDistMat()
        
        # Q-Learning Parameters
        self.num_nodes = self.data.n
        self.alpha = alpha          # Learning Rate
        self.gamma = gamma          # Discount Factor
        self.epsilon = epsilon      # Exploration Rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-Table [Current Node][Next Node]
        self.q_table = np.zeros((self.num_nodes, self.num_nodes))
        
        # Environment Settings
        self.speed = 1.0
        # Determine maximum possible capacity across all available ships
        # (Used to know when we MUST return to depot)
        self.max_fleet_capacity = max([ship['Capacity'] for ship in self.data.ships])

    def get_valid_actions(self, current_node, unvisited, current_load, current_time):
        """
        Returns a list of node IDs that are valid to visit next based on:
        1. Remaining unvisited set.
        2. Vehicle Capacity (using max fleet cap as upper bound).
        3. Time Windows.
        """
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
            
            # If arrival is after the deadline (end of TW), it's invalid
            # Note: time_windows[node][1] can be float('inf')
            if arrival_time <= self.data.time_windows[node][1]:
                valid_nodes.append(node)
                
        return valid_nodes

    def choose_action(self, current_node, valid_nodes):
        # Exploration
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_nodes)
        
        # Exploitation
        # Find max Q-value among valid nodes only
        q_values = {node: self.q_table[current_node][node] for node in valid_nodes}
        max_q = max(q_values.values())
        
        # Handle ties randomly
        best_candidates = [node for node, q in q_values.items() if q == max_q]
        return random.choice(best_candidates)

    def learn(self, state, action, reward, next_state, possible_future_nodes):
        old_value = self.q_table[state][action]
        
        if possible_future_nodes:
            future_max = max([self.q_table[next_state][n] for n in possible_future_nodes])
        else:
            future_max = 0 # Terminal (Depot)
            
        # Bellman Equation
        new_value = old_value + self.alpha * (reward + self.gamma * future_max - old_value)
        self.q_table[state][action] = new_value

    def assign_vehicle(self, route_load):
        """
        Finds the cheapest vehicle that can hold the specific route_load.
        """
        best_vehicle = None
        min_rent = float('inf')
        
        for ship in self.data.ships:
            if ship['Capacity'] >= route_load:
                if ship['Rent'] < min_rent:
                    min_rent = ship['Rent']
                    best_vehicle = ship
                    
        if best_vehicle is None:
            # Fallback (Should not happen if max_fleet_capacity logic works)
            best_vehicle = self.data.ships[-1] 
            
        return best_vehicle

    def solve(self, episodes=2000):
        print(f"Training Q-Learning Agent over {episodes} episodes...")
        
        best_global_dist = float('inf')
        best_routes_solution = []

        for ep in range(episodes):
            unvisited = set(self.data.ids)
            unvisited.remove(0) # Remove Depot
            
            routes = []
            total_dist_episode = 0
            
            # While there are customers left to visit
            while unvisited:
                current_node = 0 # Start at Depot
                current_load = 0
                current_time = self.data.time_windows[0][0] # Depot start time
                route = [] 
                
                while True:
                    # Get valid next steps
                    valid_nodes = self.get_valid_actions(current_node, unvisited, current_load, current_time)
                    
                    if not valid_nodes:
                        # Must return to Depot
                        next_node = 0
                    else:
                        # Choose based on Q-Table
                        next_node = self.choose_action(current_node, valid_nodes)
                    
                    # Calculate specifics
                    dist = self.data.dist_matrix[current_node][next_node]
                    
                    # REWARD FUNCTION: Negative distance (minimize distance)
                    reward = -dist 
                    
                    if next_node != 0:
                        # Update State
                        travel_time = dist / self.speed
                        arrival_time = max(current_time + travel_time, self.data.time_windows[next_node][0])
                        service_time = self.data.service_times[next_node]
                        
                        # Learn
                        # Check what would be valid from the NEXT node to update Q-value
                        future_unvisited = unvisited.copy()
                        future_unvisited.remove(next_node)
                        future_valid = self.get_valid_actions(next_node, future_unvisited, 
                                                              current_load + self.data.demands[next_node], 
                                                              arrival_time + service_time)
                        
                        self.learn(current_node, next_node, reward, next_node, future_valid)
                        
                        # Commit move
                        current_node = next_node
                        current_load += self.data.demands[next_node]
                        current_time = arrival_time + service_time
                        unvisited.remove(next_node)
                        route.append(current_node)
                        total_dist_episode += dist
                        
                    else:
                        # Returning to depot
                        self.learn(current_node, 0, reward, 0, [])
                        total_dist_episode += dist
                        break # End of this route
                
                routes.append(route)
                
            # Decay Epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Track best solution found during training
            if total_dist_episode < best_global_dist:
                best_global_dist = total_dist_episode
                best_routes_solution = routes

        self.print_solution(best_routes_solution)

    def print_solution(self, routes):
        total_hired_cost = 0
        total_distance = 0
        total_load = 0
        
        print("\n" + "="*60)
        print(f"{'VEHICLE USAGE & PATHS (Q-Learning)':^60}")
        print("="*60)

        for i, route_node_indices in enumerate(routes):
            # 1. Calculate Route Metrics
            route_load = sum(self.data.demands[n] for n in route_node_indices)
            
            # 2. Assign Best Vehicle
            vehicle = self.assign_vehicle(route_load)
            total_hired_cost += vehicle['Rent']
            total_load += route_load
            
            # 3. Calculate precise distance and path string
            curr = 0 # Depot
            dist = 0
            path_str = [str(self.data.ids[0])]
            
            for node in route_node_indices:
                dist += self.data.dist_matrix[curr][node]
                curr = node
                path_str.append(str(self.data.ids[node]))
                
            # Return to depot
            dist += self.data.dist_matrix[curr][0]
            path_str.append(str(self.data.ids[0]))
            
            total_distance += dist
            
            print(f"\n[Vehicle ID: {i}]")
            print(f"   Type     : {vehicle['Type']}")
            print(f"   Rent     : ${vehicle['Rent']:,.0f}")
            print(f"   Capacity : {vehicle['Capacity']} tons")
            print(f"   Load     : {route_load:.2f} tons")
            print(f"   Distance : {dist:.2f} km")
            print(f"   Path     : {' -> '.join(path_str)}")

        print("\n" + "="*60)
        print(f"{'FINAL SUMMARY':^60}")
        print("="*60)
        print(f"Total Hired Cost (Rent) : ${total_hired_cost:,.0f}")
        print(f"Total Distance Travelled: {total_distance:.2f} km")
        print(f"Total Cargo Delivered   : {total_load:.2f} tons")
        print("="*60)

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    csv_path = "datasets/marine_debris_20.csv"
    
    try:
        data_obj = Data(csv_path)
        # You can increase episodes if results are poor (RL takes time to converge)
        agent = QLearningSolver(data_obj, alpha=0.1, gamma=0.9) 
        agent.solve(episodes=3000)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")