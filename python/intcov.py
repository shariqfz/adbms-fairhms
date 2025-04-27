import sys
import math
import time
from collections import defaultdict, namedtuple
from typing import List, Dict, Tuple, Set

Element = namedtuple('Element', 
    ['idx', 'x', 'y', 'slope', 'intercept', 'group', 'slope_rightest', 'dis_rightest']
)
EnvelopeElement = namedtuple('EnvelopeElement', 
                           ['x', 'y', 'slope', 'intercept', 'Lx', 'Ly', 'Rx', 'Ry'])
Cover = namedtuple('Cover', ['idx', 'L', 'R'])
EPSILON = 1e-9

class IntervalCover:
    def __init__(self):
        self.elements: List[Element] = []
        self.groups: Dict[str, int] = {}
        self.convex_hull: List[Element] = []
        self.envelope: List[EnvelopeElement] = []
        self.tau_candidates: List[float] = []
        self.acc = 1e-9
        
    def read_data(self, file_path: str, group_cols: List[int]) -> None:
        """Read 2D data from file with group columns"""
        with open(file_path, 'r') as f:
            next(f)  # Skip header
            idx = 1
            for line in f:
                parts = line.strip().split('\t')
                x = float(parts[0])
                y = float(parts[1])
                group = self._get_group(parts, group_cols)
                
                slope = (x - y) / 1.0  # Assuming range_dis=1
                intercept = y
                
                # Add default values for new fields
                self.elements.append(Element(
                    idx=idx,
                    x=x,
                    y=y,
                    slope=slope,
                    intercept=intercept,
                    group=group,
                    slope_rightest=0.0,  # Initialize with default
                    dis_rightest=0.0      # Initialize with default
                ))
                idx += 1

                
    def _get_group(self, parts: List[str], group_cols: List[int]) -> int:
        """Convert group columns to unique group ID"""
        group_key = ''.join(parts[i] for i in group_cols)
        if group_key not in self.groups:
            self.groups[group_key] = len(self.groups)
        return self.groups[group_key]

    def preprocess(self):
        """Matches C++ struct element fields"""
        if not self.elements:
            return
            
        # Find max_x and max_y as in C++
        self.max_x = max(self.elements, key=lambda e: (e.x, e.y))
        self.max_y = max(self.elements, key=lambda e: (e.y, e.x))
        
        # Calculate rightest slopes with proper field names
        processed = []
        for e in self.elements:
            dx = self.max_x.x - e.x
            dy = self.max_x.y - e.y
            if abs(dx) < EPSILON:
                slope_rightest = 0.0
                dis_rightest = 0.0
            else:
                slope_rightest = dy / dx
                dis_rightest = dx**2 + dy**2
                
            # Create new Element with all required fields
            processed.append(e._replace(
                slope_rightest=slope_rightest,
                dis_rightest=dis_rightest
            ))
            
        self.elements = sorted(processed, key=lambda e: (e.slope_rightest, -e.dis_rightest))


    def build_envelope(self) -> None:
        """Construct convex hull and envelope"""
        if not self.elements:
            return
            
        hull = [self.max_x]
        for e in self.elements:
            if e.y <= hull[-1].y:
                continue
                
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                slope = (a.y - b.y) / (a.x - b.x)
                intercept = a.y - slope * a.x
                
                if slope * e.x + intercept > e.y:
                    break
                hull.pop()
                
            hull.append(e)
            if e.y == self.max_y.y:
                break
                
        self.convex_hull = hull[::-1]
        self._create_envelope_segments()

    def _create_envelope_segments(self) -> None:
        """Create envelope segments from convex hull"""
        if not self.convex_hull:
            return
            
        envelope = []
        prev = self.convex_hull[0]
        for e in self.convex_hull[1:]:
            Lx = 0.0
            Ly = prev.intercept
            Rx = (e.intercept - prev.intercept) / (prev.slope - e.slope)
            Ry = prev.slope * Rx + prev.intercept
            
            envelope.append(EnvelopeElement(
                x=prev.x, y=prev.y, slope=prev.slope, intercept=prev.intercept,
                Lx=Lx, Ly=Ly, Rx=Rx, Ry=Ry
            ))
            prev = e
            
        # Add final segment
        envelope.append(EnvelopeElement(
            x=prev.x, y=prev.y, slope=prev.slope, intercept=prev.intercept,
            Lx=Rx, Ly=Ry, Rx=1.0, Ry=prev.x
        ))
        
        self.envelope = envelope

    def calculate_tau_candidates(self) -> None:
        """Calculate potential tau values for binary search"""
        intersections = []
        for e in self.elements:
            intersections.extend([(0.0, e.x), (1.0, e.y)])
            
        # Calculate pairwise intersections
        for i in range(len(self.elements)):
            for j in range(i+1, len(self.elements)):
                e1 = self.elements[i]
                e2 = self.elements[j]
                
                denom = e2.slope - e1.slope
                if abs(denom) < EPSILON:
                    continue
                    
                x = (e1.intercept - e2.intercept) / denom
                if 0 <= x <= 1.0:
                    y = e1.slope * x + e1.intercept
                    intersections.append((x, y))
                    
        # Sort and calculate tau values
        intersections.sort()
        self.tau_candidates = []
        for x, y in intersections:
            for seg in self.envelope:
                if seg.Lx - EPSILON <= x <= seg.Rx + EPSILON:
                    envelope_y = seg.slope * x + seg.intercept
                    if envelope_y > 0:
                        self.tau_candidates.append(y / envelope_y)
                    break
                    
        self.tau_candidates = sorted(list(set(self.tau_candidates)))

    def optimize_coverage(self, group_upper: List[int], group_lower: List[int], k: int) -> Tuple[float, List[int]]:
        """Main optimization workflow"""
        best_tau = 0.0
        best_path = []
        
        # Binary search on tau candidates
        low, high = 0, len(self.tau_candidates)
        while high - low > 1:
            mid = (high + low) // 2
            tau = self.tau_candidates[mid]
            coverage = self.calculate_coverage(tau, group_upper, group_lower, k)
            
            if abs(coverage[0] - 1.0) < EPSILON:
                low = mid
                best_tau, best_path = tau, coverage[1]
            else:
                high = mid
                
        return best_tau, self.post_process(best_path, group_lower)

    def calculate_coverage(self, tau: float, group_upper: List[int], group_lower: List[int], k: int) -> Tuple[float, List[int]]:
        """Calculate coverage for given tau with DP"""
        # Implementation of coverage calculation and DP stack
        # ... (detailed implementation would go here)
        return 1.0, []  # Placeholder

    def post_process(self, path: List[int], group_lower: List[int]) -> List[int]:
        """Add points to satisfy group constraints"""
        # Implementation of post-processing
        return path  # Placeholder

def run_intcov(file_path: str, group_cols: List[int], 
              upper_bounds: List[int], lower_bounds: List[int], 
              k: int) -> Tuple[List[int], float]:
    """Main entry point for interval covering"""
    ic = IntervalCover()
    start = time.time()
    
    try:
        ic.read_data(file_path, group_cols)
        ic.preprocess()
        ic.build_envelope()
        ic.calculate_tau_candidates()
        tau, result = ic.optimize_coverage(upper_bounds, lower_bounds, k)
        return result, time.time() - start
    except Exception as e:
        print(f"Error processing intcov: {str(e)}")
        return [], 0.0