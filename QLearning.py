import pandas as pd
import numpy as np
import math
import random
import re
import IOReader as IO

# --- Q-LEARNING AGENT ---
class QAgent:
    def __init__(self, num_states, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.q_table = np.zeros((num_states, num_states)) # State: CurrentNode -> Action: NextNode
        self.alpha = alpha      # Learning Rate
        self.gamma = gamma      # Discount Factor
        self.epsilon = epsilon  # Exploration Rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01

    def choose_action(self, current_node, valid_next_nodes):
        if not valid_next_nodes:
            return 0 # Return to depot if no valid nodes
        
        # Exploration
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_next_nodes)
        
        # Exploitation: Choose node with highest Q value
        # We only look at Q-values for valid next nodes
        q_values = {node: self.q_table[current_node][node] for node in valid_next_nodes}
        max_q = max(q_values.values())
        
        # Handle ties randomly
        best_nodes = [node for node, q in q_values.items() if q == max_q]
        return random.choice(best_nodes)

    def learn(self, state, action, reward, next_state, possible_future_actions):
        old_value = self.q_table[state][action]
        
        if possible_future_actions:
            future_max = max([self.q_table[next_state][n] for n in possible_future_actions])
        else:
            future_max = 0 # Terminal state (end of route)
            
        # Bellman Equation
        new_value = old_value + self.alpha * (reward + self.gamma * future_max - old_value)
        self.q_table[state][action] = new_value

    def decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    # --- 3. ENVIRONMENT & TRAINING LOGIC ---
    def solve(self, data_obj, episodes=1000):
        """
        Runs the training loop using the internal Q-table and parameters.
        """
        best_total_dist = float('inf')
        best_routes = []

        for ep in range(episodes):
            unvisited = set(range(1, data_obj.num_nodes)) # All nodes except depot (0)
            routes = []
            total_dist = 0
            
            # Start a new set of routes until all customers visited
            while unvisited:
                curr_node = 0 # Start at Depot
                curr_load = 0
                curr_time = 0 
                route = [0]
                
                while True:
                    # 1. Identify Valid Actions (Constraints check)
                    valid_actions = []
                    for cand in unvisited:
                        dist = data_obj.dist_mat[curr_node][cand]
                        travel_time = dist / data_obj.speed
                        arrival_time = max(curr_time + travel_time, data_obj.nodes[cand]['tw_start'])
                        
                        # Check Capacity & Time Window End
                        if (curr_load + data_obj.nodes[cand]['demand'] <= data_obj.vehicle_cap) and \
                           (arrival_time <= data_obj.nodes[cand]['tw_end']):
                            valid_actions.append(cand)
                    
                    # 2. Agent chooses action
                    if not valid_actions:
                        next_node = 0 
                    else:
                        next_node = self.choose_action(curr_node, valid_actions)

                    # 3. Execute Step
                    dist = data_obj.dist_mat[curr_node][next_node]
                    reward = -dist * 10 
                    
                    # Update State
                    if next_node != 0:
                        travel_time = dist / data_obj.speed
                        arrival_time = max(curr_time + travel_time, data_obj.nodes[next_node]['tw_start'])
                        curr_time = arrival_time + data_obj.nodes[next_node]['service_time']
                        curr_load += data_obj.nodes[next_node]['demand']
                        
                        unvisited.remove(next_node)
                        route.append(next_node)
                        
                        # Look ahead for learning
                        future_actions = [n for n in unvisited] 
                        self.learn(curr_node, next_node, reward, next_node, future_actions)
                        
                        curr_node = next_node
                    else:
                        # Returning to Depot
                        route.append(0)
                        self.learn(curr_node, 0, reward, 0, []) 
                        break 
                
                routes.append(route)
                # Calculate route distance
                r_dist = 0
                for i in range(len(route)-1):
                    r_dist += data_obj.dist_mat[route[i]][route[i+1]]
                total_dist += r_dist

            # Track best solution
            if total_dist < best_total_dist:
                best_total_dist = total_dist
                best_routes = routes
                
            self.decay()

        return best_routes, best_total_dist

# --- 4. EXECUTION ---
# Run Solver
data = Data(raw_data)
print(f"Training Q-Learning Agent on {data.num_nodes} nodes...")
routes, dist = solve_vrptw(data, episodes=2000)

print("\n--- Optimized Routes ---")
print(f"Total Distance: {dist:.2f} km")
for i, route in enumerate(routes):
    print(f"Vehicle {i+1}: {route} (Load: {sum([data.nodes[n]['demand'] for n in route]):.2f} tons)")