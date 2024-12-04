

class Idea:
    def __init__(
        self, 
        seed: str, 
        depth: int = 0, 
        lineage: list["Idea"] = [],
        abs_rating: int = 0
    ):
        self.seed = seed
        self.depth = depth
        self.lineage = lineage.copy()
        
        # Content
        self.content = ''
        
        # Ratings
        self.abs_rating = abs_rating # absolute rating
        self.elo_rating = 1000 # elo rating
    

    def __lt__(self, other: "Idea") -> bool:
        
        return self.abs_rating < other.abs_rating

    def __eq__(self, other: "Idea") -> bool:
        
        return self.abs_rating == other.abs_rating
