
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import cv2
import io
import sys
import os

# Add current directory to path to ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rc_network import RCNetworkModel
from visual_complexity import VisualComplexityAnalyzer
from fractal_dimension import FractalAnalyzer

app = FastAPI(title="AHI 2.0 Analysis Backend")

# --- Pydantic Models ---

class RoomParameters(BaseModel):
    wall_area: float
    window_area: float
    floor_area: float
    volume: float
    wall_thickness: float = 0.3
    u_value_wall: float = 0.3
    u_value_window: float = 1.2
    air_change_rate: float = 0.5
    # Optional: material properties override
    wall_density: Optional[float] = 2400
    
class ThermalAnalysisRequest(BaseModel):
    room: RoomParameters
    external_temperature: float = 0.0 # Current outdoor temp
    solar_radiation: float = 0.0      # Current solar rad

# --- Analyzers ---
visual_analyzer = VisualComplexityAnalyzer()
fractal_analyzer = FractalAnalyzer()

@app.get("/")
def read_root():
    return {"status": "AHI 2.0 Backend Running"}

@app.post("/analyze/thermal")
def analyze_thermal(request: ThermalAnalysisRequest):
    """
    Analyze thermal state of a room and return boundary conditions.
    Uses RC-Network model (5R1C).
    """
    try:
        # Convert request to dictionary for RCNetworkModel
        params = request.room.dict()
        
        # Add material dict
        params['wall_material'] = {
            'density': request.room.wall_density,
            'specific_heat': 1000, # Default
            'conductivity': 1.7    # Default
        }
        
        # Initialize model
        model = RCNetworkModel(params)
        
        # Run a quick "settling" simulation or just use current state?
        # For this endpoint, we want the steady state response to current conditions
        # OR we update the model with current conditions.
        # Since RCNetwork is stateful but we recreate it per request (stateless API),
        # we should simulate a period to reach equilibrium or 24h.
        # Let's simulate 24h with CONSTANT conditions to find steady state.
        
        weather_data = {
            'temperature': [request.external_temperature] * 24,
            'solar_radiation': [request.solar_radiation] * 24
        }
        
        results = model.simulate_period(weather_data)
        
        # Get boundary conditions from the end of simulation
        bc = model.export_boundary_conditions(results)
        
        return {
            "boundary_conditions": bc,
            "thermal_properties": model.get_thermal_properties(),
            "summary": results['summary']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/aesthetics")
async def analyze_aesthetics(file: UploadFile = File(...)):
    """
    Analyze aesthetic properties of an image (Visual Complexity & Fractals).
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        # 1. Visual Complexity Analysis
        complexity_metrics = visual_analyzer.analyze(image)
        
        # 2. Fractal Analysis
        fractal_metrics = fractal_analyzer.analyze_image(image)
        
        # 3. Combined Report
        return {
            "complexity": {
                "shannon_entropy": complexity_metrics.shannon_entropy,
                "overall_complexity": complexity_metrics.overall_complexity,
                "color_harmony": complexity_metrics.color_harmony_score,
                "interpretation": complexity_metrics.interpretation
            },
            "fractal": {
                "dimension": fractal_metrics.dimension,
                "stress_level": fractal_metrics.stress_level,
                "interpretation": fractal_metrics.interpretation
            },
            "ahi_score": (complexity_metrics.overall_complexity + 
                          (1.0 if fractal_metrics.stress_level == 'optimal' else 0.5)) / 2.0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
