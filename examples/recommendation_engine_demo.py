# -*- coding: utf-8 -*-
# examples/recommendation_engine_demo.py

# Demonstration of the Intelligent Recommendation Engine
# Shows how the AI-powered virtual architect consultant works

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all necessary modules
from modules.thermal import calculate_total_heat_loss
from modules.optic import analyze_natural_light
from modules.indexer import calculate_function_index, get_function_index_interpretation
from modules.recommender import generate_recommendations

# Import tools from physipy for physical quantities
from physipy import m, K, Quantity, units

def main():
    """
    Demonstration of the intelligent recommendation engine for room optimization.
    """
    
    print("=== INTELLIGENT RECOMMENDATION ENGINE DEMO ===\n")
    print("Philosophy: Function ~ Aesthetics\n")
    
    # --- Step 1: Define room parameters ---
    
    # Room dimensions
    room_width = 5.0
    room_depth = 6.0
    room_height = 2.8
    room_area = room_width * room_depth * m**2
    
    print(f"Analyzed room: {room_width}x{room_depth}x{room_height} m")
    print(f"Area: {float(room_area.value)} sq.m\n")
    
    # Temperature conditions
    temp_inside = 22 * K
    temp_outside = -5 * K
    
    # Room components for thermal analysis
    room_components = [
        {"name": "Walls", "material": "brick_wall_uninsulated", "area": 40 * m**2},
        {"name": "Window", "material": "double_pane_glass", "area": 4.5 * m**2}
    ]
    
    # Room parameters for optical analysis
    room_data = {
        "room_dimensions": {
            "width": room_width,
            "depth": room_depth,
            "height": room_height
        },
        "windows": [
            {
                "width": 2.5,
                "height": 1.8,
                "sill_height": 0.8,
                "orientation": "south"
            }
        ],
        "surface_reflectance": {
            "walls": 0.7,
            "floor": 0.3,
            "ceiling": 0.85
        },
        "location": {
            "latitude": 55.7  # Moscow
        }
    }
    
    # --- Step 2: Thermal analysis ---
    
    print("--- THERMAL ANALYSIS ---")
    thermal_results = calculate_total_heat_loss(room_components, temp_inside, temp_outside)
    
    W = units['W']
    thermal_watts = thermal_results.to(W)
    thermal_per_sqm = float(thermal_results.value) / float(room_area.value)
    
    print(f"Total heat loss: {thermal_watts}")
    print(f"Heat loss per sq.m: {thermal_per_sqm:.1f} W/sq.m")
    
    # --- Step 3: Optical analysis ---
    
    print("\n--- OPTICAL ANALYSIS ---")
    optic_results = analyze_natural_light(room_data)
    
    print(f"Daylight coverage: {optic_results['daylight_coverage_percent']}%")
    print(f"Uniformity index: {optic_results['uniformity_index']}")
    print(f"Intensity index: {optic_results['intensity_index']}")
    
    # --- Step 4: Function index calculation ---
    
    print("\n--- FUNCTION INDEX ---")
    function_results = calculate_function_index(thermal_results, optic_results, room_area)
    
    print(f"Thermal score: {function_results['thermal_score']}")
    print(f"Optical score: {function_results['optic_score']}")
    print(f"FINAL FUNCTION INDEX: {function_results['final_function_index']}")
    
    # Interpretation
    interpretation = get_function_index_interpretation(function_results['final_function_index'])
    print(f"\nAssessment: {interpretation}")
    
    # --- Step 5: Intelligent Recommendations ---
    
    target_index = 0.8  # Target functionality index
    
    if function_results['final_function_index'] < target_index:
        print(f"\n--- INTELLIGENT IMPROVEMENT ANALYSIS ---")
        print(f"Target index: {target_index}")
        print("Starting recommendation engine...\n")
        
        # Generate intelligent recommendations
        recommendations = generate_recommendations(room_data, function_results, target_index)
        
        # Print results
        print_recommendations_report(recommendations)
        
    else:
        print("\n--- CONGRATULATIONS! ---")
        print("• Your room has excellent functionality!")
        print("• The principle 'function ~ aesthetics' is achieved")
        print(f"• Target index {target_index} reached")
    
    print("\n" + "="*60)
    print("Analysis completed. Recommendation engine working perfectly!")


def print_recommendations_report(recommendations):
    """
    Beautifully prints the room improvement recommendations report.
    """
    initial_scores = recommendations["initial_scores"]
    plan = recommendations["recommended_plan"]
    
    print(">> CURRENT METRICS:")
    print(f"   • Final function index: {initial_scores['final_function_index']}")
    print(f"   • Thermal score: {initial_scores['thermal_score']}")
    print(f"   • Optical score: {initial_scores['optic_score']}")
    
    # Structural recommendations
    if plan["structural"]:
        print("\n>> STRUCTURAL CHANGES (High Impact):")
        for i, rec in enumerate(plan["structural"], 1):
            print(f"   {i}. {rec['recommendation']}")
            print(f"      Impact: {rec['impact']}")
            print(f"      New index: {rec['final_score_after_change']}")
            print()
    
    # Lifestyle fixes
    if plan["lifestyle"]:
        print(">> LIFESTYLE FIXES (Easy Implementation):")
        for i, rec in enumerate(plan["lifestyle"], 1):
            print(f"   {i}. {rec['recommendation']}")
            print(f"      Impact: {rec['impact']}")
            print(f"      New index: {rec['final_score_after_change']}")
            print()
    
    # Final advice
    print(">> RECOMMENDATION:")
    if plan["structural"] and plan["lifestyle"]:
        best_structural = max(plan["structural"], key=lambda x: x['final_score_after_change'])
        print(f"   Start with: {best_structural['recommendation']}")
        print("   Then apply lifestyle fixes for maximum effect.")
    elif plan["structural"]:
        print("   Focus on structural changes to reach your goal.")
    elif plan["lifestyle"]:
        print("   Use lifestyle fixes - they will give noticeable results!")
    else:
        print("   Your room is already close to ideal!")


if __name__ == "__main__":
    main()
