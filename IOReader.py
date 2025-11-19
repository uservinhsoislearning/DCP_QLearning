import pandas as pd
import math

class Data:
    def __init__(self, filepath: str):
        """
        Reads the CSV file and initializes data structures.
        """
        self.filepath = filepath
        
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file at {filepath}")

        # Extracting columns
        self.ids = df['no.'].tolist()
        self.coords = list(zip(df['latitude(N)'], df['longitude(E)']))
        self.demands = df['weight(tons)'].tolist()
        self.service_times = df['collection_time(h)'].tolist()
        
        # Parse time windows
        self.time_windows = []
        for tw in df['time_windows']:
            try:
                start, end = map(float, str(tw).split('-'))
                self.time_windows.append((start, end))
            except ValueError:
                self.time_windows.append((0.0, float('inf')))

        # ------------------------------------------
        # ADD DEPOT NODE (hard-coded)
        # ------------------------------------------
        depot_id = 0
        depot_coord = (50.0, 0.0)   # Example depot coordinate
        depot_demand = 0
        depot_service_time = 0
        depot_time_window = (0.0, float('inf'))

        # Insert depot at the beginning
        self.ids.insert(0, depot_id)
        self.coords.insert(0, depot_coord)
        self.demands.insert(0, depot_demand)
        self.service_times.insert(0, depot_service_time)
        self.time_windows.insert(0, depot_time_window)

        # Update number of nodes
        self.n = len(self.ids)

        # Placeholder for distance matrix
        self.dist_matrix = []

        # ------------------------------------------
        # SHIP INFORMATION
        # ------------------------------------------
        self.ships = [
            {"Type": "T1",  "Rent": 10000, "Capacity": 7},
            {"Type": "T2", "Rent": 11000, "Capacity": 8},
            {"Type": "T3",  "Rent": 12000, "Capacity": 9},
            {"Type": "T4",  "Rent": 13000, "Capacity": 10},
            {"Type": "T5", "Rent": 14000, "Capacity": 11},
            {"Type": "T6",  "Rent": 15000, "Capacity": 12},
            {"Type": "T7",  "Rent": 16000, "Capacity": 13},
            {"Type": "T8", "Rent": 17000, "Capacity": 14}
        ]

    def _euclidean(self, coord1, coord2):
        """
        Calculates the great-circle distance between two points 
        on the Earth using the Euclidean formula.
        Input: (lat, lon) tuples.
        Output: Distance.
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        return math.sqrt((lat1-lat2)**2+(lon1-lon2)**2)

    def calcDistMat(self):
        """
        Calculates a symmetric N x N distance matrix.
        """
        # Initialize N x N matrix with zeros
        self.dist_matrix = [[0.0] * self.n for _ in range(self.n)]
        
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    self.dist_matrix[i][j] = 0.0
                else:
                    dist = self._euclidean(self.coords[i], self.coords[j])
                    self.dist_matrix[i][j] = round(dist, 2) # Rounding to 3 decimal places
        
        return self.dist_matrix

# --- Testing block ---
# if __name__ == "__main__":
#     data = Data("datasets\marine_debris_20.csv")
#     print(f"Data:\n{data.ids}\n{data.coords}\n{data.demands}\n{data.service_times}\n{data.n}\n{data.dist_matrix}")
#     data.calcDistMat()
#     print(f"Distance Matrix:\n{data.dist_matrix}")