"""
Visual Complexity Analysis Module
Measures information entropy, color harmony, and perceptual complexity
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2
from scipy import stats, spatial
from sklearn.cluster import KMeans
import colorsys


@dataclass
class ComplexityMetrics:
    """Container for visual complexity analysis results"""
    shannon_entropy: float  # Information entropy (0-8 bits)
    edge_entropy: float  # Edge distribution entropy
    color_entropy: float  # Color diversity entropy
    color_harmony_score: float  # Color scheme harmony (0-1)
    spatial_frequency: float  # High-frequency content
    overall_complexity: float  # Combined metric (0-1)
    interpretation: str


class VisualComplexityAnalyzer:
    """
    Analyzes visual complexity using information theory and perceptual metrics
    
    Based on research showing that moderate complexity (not too simple, not too chaotic)
    is most aesthetically pleasing and psychologically comfortable
    """
    
    # Optimal entropy ranges (in bits)
    OPTIMAL_ENTROPY_RANGE = (4.0, 6.0)  # For 8-bit images
    
    # Color harmony schemes
    COLOR_SCHEMES = {
        'complementary': 180,  # Opposite on color wheel
        'analogous': 30,  # Adjacent colors
        'triadic': 120,  # Three equidistant colors
        'split_complementary': 150,  # Base + two adjacent to complement
        'tetradic': 90  # Four colors in rectangle
    }
    
    def __init__(self):
        """Initialize the visual complexity analyzer"""
        self.last_analysis = None
    
    def analyze(self, image: np.ndarray) -> ComplexityMetrics:
        """
        Perform comprehensive visual complexity analysis
        
        Args:
            image: Input image (RGB or grayscale)
        
        Returns:
            ComplexityMetrics object with analysis results
        """
        # Calculate various entropy measures
        shannon_entropy = self._calculate_shannon_entropy(image)
        edge_entropy = self._calculate_edge_entropy(image)
        
        # Color analysis (if RGB)
        if len(image.shape) == 3:
            color_entropy = self._calculate_color_entropy(image)
            color_harmony = self._analyze_color_harmony(image)
        else:
            color_entropy = shannon_entropy  # Use grayscale entropy
            color_harmony = 0.5  # Neutral for grayscale
        
        # Spatial frequency analysis
        spatial_freq = self._calculate_spatial_frequency(image)
        
        # Calculate overall complexity score
        overall = self._calculate_overall_complexity(
            shannon_entropy, edge_entropy, color_entropy, spatial_freq
        )
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            shannon_entropy, color_harmony, overall
        )
        
        metrics = ComplexityMetrics(
            shannon_entropy=shannon_entropy,
            edge_entropy=edge_entropy,
            color_entropy=color_entropy,
            color_harmony_score=color_harmony,
            spatial_frequency=spatial_freq,
            overall_complexity=overall,
            interpretation=interpretation
        )
        
        self.last_analysis = metrics
        return metrics
    
    def _calculate_shannon_entropy(self, image: np.ndarray) -> float:
        """
        Calculate Shannon entropy H = -Σ p_i log₂(p_i)
        
        Higher entropy = more information/complexity
        Lower entropy = more predictable/simple
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate histogram
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        
        # Normalize to get probabilities
        hist = hist.astype(float)
        hist = hist / hist.sum()
        
        # Remove zero entries
        hist = hist[hist > 0]
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        return entropy
    
    def _calculate_edge_entropy(self, image: np.ndarray) -> float:
        """
        Calculate entropy of edge distribution
        Measures how edges are distributed spatially
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Divide image into blocks
        block_size = 32
        h, w = edges.shape
        blocks = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = edges[i:i+block_size, j:j+block_size]
                edge_density = np.sum(block > 0) / (block_size * block_size)
                blocks.append(edge_density)
        
        if not blocks:
            return 0.0
        
        # Calculate entropy of edge distribution
        hist, _ = np.histogram(blocks, bins=10, range=(0, 1))
        hist = hist.astype(float)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        
        entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
        
        return entropy
    
    def _calculate_color_entropy(self, image: np.ndarray) -> float:
        """
        Calculate color distribution entropy
        """
        if len(image.shape) != 3:
            return self._calculate_shannon_entropy(image)
        
        # Quantize colors to reduce space
        n_colors = 64
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3)
        
        # Sample if image is too large
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        # Cluster colors
        kmeans = KMeans(n_clusters=min(n_colors, len(np.unique(pixels, axis=0))), 
                        random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Calculate entropy of color distribution
        unique, counts = np.unique(labels, return_counts=True)
        probs = counts.astype(float) / counts.sum()
        
        entropy = -np.sum(probs * np.log2(probs))
        
        return entropy
    
    def _analyze_color_harmony(self, image: np.ndarray) -> float:
        """
        Analyze color harmony based on color wheel relationships
        Returns score 0-1 (1 = perfect harmony)
        """
        if len(image.shape) != 3:
            return 0.5  # Neutral for grayscale
        
        # Extract dominant colors
        dominant_colors = self._extract_dominant_colors(image, n_colors=5)
        
        if len(dominant_colors) < 2:
            return 0.5  # Not enough colors to analyze
        
        # Convert to HSV
        hsv_colors = []
        for color in dominant_colors:
            rgb = color / 255.0
            hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
            hsv_colors.append(hsv)
        
        # Analyze hue relationships
        hues = [c[0] * 360 for c in hsv_colors]
        harmony_score = self._evaluate_color_scheme(hues)
        
        # Factor in saturation and value consistency
        saturations = [c[1] for c in hsv_colors]
        values = [c[2] for c in hsv_colors]
        
        sat_consistency = 1 - np.std(saturations)
        val_consistency = 1 - np.std(values)
        
        # Combined harmony score
        final_score = (harmony_score * 0.6 + 
                      sat_consistency * 0.2 + 
                      val_consistency * 0.2)
        
        return np.clip(final_score, 0, 1)
    
    def _extract_dominant_colors(self, image: np.ndarray, n_colors: int = 5) -> List[np.ndarray]:
        """Extract dominant colors using k-means clustering"""
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3)
        
        # Sample if too large
        if len(pixels) > 5000:
            indices = np.random.choice(len(pixels), 5000, replace=False)
            pixels = pixels[indices]
        
        # Cluster
        n_clusters = min(n_colors, len(np.unique(pixels, axis=0)))
        if n_clusters < 2:
            return [pixels[0]]
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Sort by cluster size
        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        
        dominant_colors = [kmeans.cluster_centers_[i] for i in sorted_indices]
        
        return dominant_colors
    
    def _evaluate_color_scheme(self, hues: List[float]) -> float:
        """
        Evaluate how well colors fit standard harmony schemes
        Returns score 0-1
        """
        if len(hues) < 2:
            return 0.5
        
        best_score = 0
        
        # Check each harmony scheme
        for scheme_name, expected_angle in self.COLOR_SCHEMES.items():
            score = 0
            comparisons = 0
            
            for i in range(len(hues)):
                for j in range(i + 1, len(hues)):
                    angle_diff = abs(hues[i] - hues[j])
                    angle_diff = min(angle_diff, 360 - angle_diff)  # Wrap around
                    
                    # Score based on how close to expected angle
                    if scheme_name == 'analogous':
                        # Adjacent colors
                        if angle_diff <= expected_angle:
                            score += 1 - (angle_diff / expected_angle)
                    else:
                        # Other schemes need specific angles
                        deviation = abs(angle_diff - expected_angle)
                        if deviation < 30:  # Allow 30° tolerance
                            score += 1 - (deviation / 30)
                    
                    comparisons += 1
            
            if comparisons > 0:
                scheme_score = score / comparisons
                best_score = max(best_score, scheme_score)
        
        return best_score
    
    def _calculate_spatial_frequency(self, image: np.ndarray) -> float:
        """
        Calculate spatial frequency content using FFT
        High frequency = more detail/texture
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Calculate radial frequency distribution
        h, w = gray.shape
        center = (h // 2, w // 2)
        
        # Create radial bins
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r_max = min(center)
        
        # Calculate energy in high frequencies (outer 50% of spectrum)
        high_freq_mask = r > (r_max * 0.5)
        low_freq_mask = r <= (r_max * 0.5)
        
        high_freq_energy = np.sum(magnitude[high_freq_mask]**2)
        low_freq_energy = np.sum(magnitude[low_freq_mask]**2)
        
        if low_freq_energy > 0:
            freq_ratio = high_freq_energy / (high_freq_energy + low_freq_energy)
        else:
            freq_ratio = 0
        
        return freq_ratio
    
    def _calculate_overall_complexity(self, shannon: float, edge: float, 
                                     color: float, spatial: float) -> float:
        """
        Calculate overall visual complexity score (0-1)
        Combines multiple metrics with weights
        """
        # Normalize shannon entropy (typically 0-8 bits)
        shannon_norm = shannon / 8.0
        
        # Normalize edge entropy (typically 0-3 bits)
        edge_norm = edge / 3.0
        
        # Normalize color entropy (typically 0-6 bits)
        color_norm = color / 6.0
        
        # Spatial frequency is already 0-1
        
        # Weighted combination
        weights = {
            'shannon': 0.3,
            'edge': 0.2,
            'color': 0.3,
            'spatial': 0.2
        }
        
        complexity = (weights['shannon'] * shannon_norm +
                     weights['edge'] * edge_norm +
                     weights['color'] * color_norm +
                     weights['spatial'] * spatial)
        
        return np.clip(complexity, 0, 1)
    
    def _generate_interpretation(self, entropy: float, harmony: float, 
                                complexity: float) -> str:
        """Generate human-readable interpretation"""
        
        # Classify complexity level
        if complexity < 0.3:
            complexity_level = "low"
            complexity_desc = "The space appears minimalist and simple"
        elif complexity < 0.6:
            complexity_level = "moderate"
            complexity_desc = "The space has balanced visual complexity"
        else:
            complexity_level = "high"
            complexity_desc = "The space is visually complex and information-rich"
        
        # Classify entropy level
        if entropy < 4.0:
            entropy_desc = "low information content (predictable patterns)"
        elif entropy < 6.0:
            entropy_desc = "moderate information content (balanced variety)"
        else:
            entropy_desc = "high information content (diverse elements)"
        
        # Classify color harmony
        if harmony > 0.7:
            harmony_desc = "Colors are harmonious and well-coordinated"
        elif harmony > 0.4:
            harmony_desc = "Colors show moderate harmony"
        else:
            harmony_desc = "Color scheme lacks clear harmony"
        
        interpretation = (
            f"{complexity_desc}, with {entropy_desc}. "
            f"{harmony_desc}. "
        )
        
        # Add recommendations
        if complexity < 0.3:
            interpretation += (
                "Consider adding textural elements, patterns, or color variation "
                "to increase visual interest."
            )
        elif complexity > 0.7:
            interpretation += (
                "Consider simplifying some elements to reduce cognitive load "
                "and create visual breathing room."
            )
        else:
            interpretation += (
                "The complexity level is well-suited for comfortable viewing "
                "and cognitive processing."
            )
        
        return interpretation
    
    def analyze_scene_balance(self, image: np.ndarray) -> Dict:
        """
        Analyze visual balance and composition
        
        Args:
            image: Input image
        
        Returns:
            Dictionary with balance metrics
        """
        # Basic complexity analysis
        metrics = self.analyze(image)
        
        # Calculate visual weight distribution
        weight_balance = self._calculate_visual_weight_balance(image)
        
        # Rule of thirds analysis
        rule_of_thirds = self._analyze_rule_of_thirds(image)
        
        # Symmetry analysis
        symmetry = self._calculate_symmetry_scores(image)
        
        return {
            'complexity_metrics': {
                'shannon_entropy': metrics.shannon_entropy,
                'edge_entropy': metrics.edge_entropy,
                'color_harmony': metrics.color_harmony_score,
                'overall_complexity': metrics.overall_complexity
            },
            'balance_metrics': {
                'horizontal_balance': weight_balance['horizontal'],
                'vertical_balance': weight_balance['vertical'],
                'radial_balance': weight_balance['radial']
            },
            'composition': {
                'rule_of_thirds_score': rule_of_thirds,
                'symmetry_horizontal': symmetry['horizontal'],
                'symmetry_vertical': symmetry['vertical'],
                'symmetry_radial': symmetry['radial']
            },
            'interpretation': metrics.interpretation,
            'recommendations': self._generate_balance_recommendations(
                weight_balance, rule_of_thirds, symmetry
            )
        }
    
    def _calculate_visual_weight_balance(self, image: np.ndarray) -> Dict:
        """Calculate distribution of visual weight"""
        # Convert to grayscale for weight calculation
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        # Calculate center of visual mass
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        weights = gray.astype(float) / 255.0
        
        total_weight = np.sum(weights)
        if total_weight > 0:
            center_x = np.sum(x_coords * weights) / total_weight
            center_y = np.sum(y_coords * weights) / total_weight
        else:
            center_x, center_y = w/2, h/2
        
        # Calculate balance scores (0 = perfect balance, 1 = maximum imbalance)
        horizontal_balance = 1 - abs(center_x - w/2) / (w/2)
        vertical_balance = 1 - abs(center_y - h/2) / (h/2)
        
        # Radial balance (distance from center)
        radial_distance = np.sqrt((center_x - w/2)**2 + (center_y - h/2)**2)
        max_distance = np.sqrt((w/2)**2 + (h/2)**2)
        radial_balance = 1 - radial_distance / max_distance
        
        return {
            'horizontal': horizontal_balance,
            'vertical': vertical_balance,
            'radial': radial_balance,
            'center_of_mass': (center_x, center_y)
        }
    
    def _analyze_rule_of_thirds(self, image: np.ndarray) -> float:
        """Analyze how well the composition follows rule of thirds"""
        # Detect edges/features
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, 50, 150)
        
        h, w = edges.shape
        
        # Define rule of thirds lines
        h_third = h // 3
        w_third = w // 3
        
        # Check feature density near thirds lines (with tolerance)
        tolerance = min(h, w) // 20  # 5% tolerance
        
        score = 0
        total_edges = np.sum(edges > 0)
        
        if total_edges > 0:
            # Vertical lines
            for x in [w_third, 2*w_third]:
                region = edges[:, max(0, x-tolerance):min(w, x+tolerance)]
                score += np.sum(region > 0) / total_edges
            
            # Horizontal lines
            for y in [h_third, 2*h_third]:
                region = edges[max(0, y-tolerance):min(h, y+tolerance), :]
                score += np.sum(region > 0) / total_edges
        
        # Normalize to 0-1 scale
        return min(1.0, score)
    
    def _calculate_symmetry_scores(self, image: np.ndarray) -> Dict:
        """Calculate various symmetry scores"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        # Horizontal symmetry
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:h//2*2, :]
        bottom_flipped = np.flipud(bottom_half)
        
        if top_half.shape == bottom_flipped.shape:
            h_symmetry = np.corrcoef(top_half.flatten(), bottom_flipped.flatten())[0, 1]
            h_symmetry = max(0, h_symmetry)  # Ensure non-negative
        else:
            h_symmetry = 0
        
        # Vertical symmetry
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:w//2*2]
        right_flipped = np.fliplr(right_half)
        
        if left_half.shape == right_flipped.shape:
            v_symmetry = np.corrcoef(left_half.flatten(), right_flipped.flatten())[0, 1]
            v_symmetry = max(0, v_symmetry)
        else:
            v_symmetry = 0
        
        # Radial symmetry (simplified)
        center = (h//2, w//2)
        radius = min(center)
        
        # Sample points in circular pattern
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        radial_score = 0
        comparisons = 0
        
        for r in [radius//4, radius//2, 3*radius//4]:
            values = []
            for angle in angles:
                x = int(center[1] + r * np.cos(angle))
                y = int(center[0] + r * np.sin(angle))
                if 0 <= x < w and 0 <= y < h:
                    values.append(gray[y, x])
            
            if len(values) > 1:
                # Check variance (low variance = high symmetry)
                radial_score += 1 - (np.std(values) / 128)  # Normalize by half of 255
                comparisons += 1
        
        radial_symmetry = radial_score / comparisons if comparisons > 0 else 0
        
        return {
            'horizontal': h_symmetry,
            'vertical': v_symmetry,
            'radial': radial_symmetry
        }
    
    def _generate_balance_recommendations(self, balance: Dict, 
                                         thirds: float, symmetry: Dict) -> List[str]:
        """Generate composition recommendations"""
        recommendations = []
        
        # Balance recommendations
        if balance['horizontal'] < 0.6:
            recommendations.append(
                "Horizontal balance is off-center. "
                "Consider redistributing visual elements left-right."
            )
        
        if balance['vertical'] < 0.6:
            recommendations.append(
                "Vertical balance is skewed. "
                "Add visual weight to the lighter area."
            )
        
        # Rule of thirds
        if thirds < 0.3:
            recommendations.append(
                "Composition doesn't follow rule of thirds. "
                "Try placing key elements along imaginary third lines."
            )
        elif thirds > 0.7:
            recommendations.append(
                "Good use of rule of thirds in composition."
            )
        
        # Symmetry
        max_symmetry = max(symmetry.values())
        if max_symmetry > 0.8:
            recommendations.append(
                "High symmetry detected. This creates formal balance."
            )
        elif max_symmetry < 0.3:
            recommendations.append(
                "Low symmetry creates dynamic, informal balance."
            )
        
        if not recommendations:
            recommendations.append(
                "The composition shows good visual balance."
            )
        
        return recommendations


# Demo function
def demo_visual_complexity():
    """Demonstrate visual complexity analysis"""
    
    analyzer = VisualComplexityAnalyzer()
    
    print("Visual Complexity Analysis Demo")
    print("=" * 50)
    
    # Create test patterns
    
    # Pattern 1: Low complexity (solid color)
    simple = np.full((256, 256, 3), 128, dtype=np.uint8)
    simple[64:192, 64:192] = 200  # Add single square
    
    print("\n1. Simple Pattern (Minimal Complexity):")
    metrics = analyzer.analyze(simple)
    print(f"   Shannon Entropy: {metrics.shannon_entropy:.2f} bits")
    print(f"   Overall Complexity: {metrics.overall_complexity:.2f}")
    print(f"   Color Harmony: {metrics.color_harmony_score:.2f}")
    print(f"   Interpretation: {metrics.interpretation[:100]}...")
    
    # Pattern 2: Moderate complexity (gradient + pattern)
    moderate = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        moderate[i, :, 0] = i  # Red gradient
        moderate[:, i, 1] = i  # Green gradient
        if i % 32 == 0:  # Add grid
            moderate[i, :, 2] = 128
            moderate[:, i, 2] = 128
    
    print("\n2. Moderate Pattern (Balanced Complexity):")
    metrics = analyzer.analyze(moderate)
    print(f"   Shannon Entropy: {metrics.shannon_entropy:.2f} bits")
    print(f"   Overall Complexity: {metrics.overall_complexity:.2f}")
    print(f"   Color Harmony: {metrics.color_harmony_score:.2f}")
    print(f"   Interpretation: {metrics.interpretation[:100]}...")
    
    # Pattern 3: High complexity (random)
    complex_pattern = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    print("\n3. Complex Pattern (High Complexity):")
    metrics = analyzer.analyze(complex_pattern)
    print(f"   Shannon Entropy: {metrics.shannon_entropy:.2f} bits")
    print(f"   Overall Complexity: {metrics.overall_complexity:.2f}")
    print(f"   Color Harmony: {metrics.color_harmony_score:.2f}")
    print(f"   Interpretation: {metrics.interpretation[:100]}...")
    
    # Analyze balance of moderate pattern
    print("\n4. Balance Analysis of Moderate Pattern:")
    balance = analyzer.analyze_scene_balance(moderate)
    print(f"   Horizontal Balance: {balance['balance_metrics']['horizontal_balance']:.2f}")
    print(f"   Vertical Balance: {balance['balance_metrics']['vertical_balance']:.2f}")
    print(f"   Rule of Thirds Score: {balance['composition']['rule_of_thirds_score']:.2f}")
    
    return analyzer


if __name__ == "__main__":
    demo_visual_complexity()
