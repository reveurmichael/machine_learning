## Heuristcs algorithms

Here is our log folder naming convention:

logs/heuristics-bfs_{timestamp}

logs/heuristics-astar_{timestamp}

logs/heuristics-hamiltonian_{timestamp}

logs/heuristics-hamiltonian_{timestamp}  

logs/heuristics-{self.algorithm_name.lower()}_{timestamp}


let's go with the code every time:
def _setup_logging(self) -> None:
    """Setup logging directory and stats manager."""
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    self.log_dir = f"logs/heuristics-{self.algorithm_name.lower()}_{timestamp}"