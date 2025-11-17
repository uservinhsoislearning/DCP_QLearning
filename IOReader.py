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

        # Extracting columns based on your requirements
        self.ids = df['no.'].tolist()
        self.coords = list(zip(df['latitude(N)'], df['longitude(E)']))
        self.demands = df['weight(tons)'].tolist()
        self.service_times = df['collection_time(h)'].tolist()
        
        # Parsing Time Windows (Assuming format "Start-End", e.g., "8.0-12.0")
        # Returns a list of tuples: [(start, end), ...]
        self.time_windows = []
        for tw in df['time_windows']:
            try:
                # Adjust the delimiter '-' if your CSV uses something else like ':'
                start, end = map(float, str(tw).split('-')) 
                self.time_windows.append((start, end))
            except ValueError:
                # Fallback if format is invalid
                self.time_windows.append((0.0, 24.0)) 

        self.n = len(self.ids)
        self.dist_matrix = []

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