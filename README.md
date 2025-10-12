# Neighbly: Architecture Harmony Index System

**An analytical engine for calculating the Architectural Harmony Index and generating recommendations for improving living spaces.**

This project is a software implementation of the **Architectural Harmony Index** concept—a metric that quantitatively assesses how well a space conforms to the ideals of function and beauty.

![Main Architectural Harmony Index Diagram]()


## Project Philosophy

The project is based on the idea that architecture is a synthesis of science and art, function and aesthetics. A quality space is not just a collection of walls and furniture, but a balanced system that affects our well-being, productivity, and emotional state.

**Neighbly.ArchitectureHarmonyIndexSystem** is an attempt to codify the key principles of architectural harmony, transforming them from abstract concepts into measurable indicators. This tool allows for an objective audit of any room, identifying its strengths and weaknesses, and proposing scientifically-grounded ways to improve it.


## What is the Architectural Harmony Index?

The Architectural Harmony Index (from 0 to 1) is a comprehensive indicator consisting of two key components:

### 1. **The Function Index**
This assesses how efficiently and comfortably a space performs from a physical and ergonomic perspective. It is calculated based on two modules:
* **Thermodynamic Analysis:** Evaluates the heat loss of a room based on building materials, glazing area, and climatic conditions.
* **Optical Analysis:** Assesses the quality of natural light, considering the size and orientation of windows, geographical location, and surface reflectance coefficients.

### 2. **The Beauty Index**
This evaluates the aesthetic appeal and integrity of a space. In the current model, it is calculated based on three heuristics:
* **Color Harmony:** Analyzes the compatibility of the main interior colors based on classic color schemes.
* **Stylistic Unity:** Assesses the consistency of furniture with the declared style of the room.
* **Spatial Efficiency:** Analyzes the balance between occupied and free space, preventing both "emptiness" and "clutter."


## Key Features

* **Comprehensive Analysis:** Calculation of over 10 different metrics, from heat loss in Watts to the color harmony index.
* **Intelligent Recommendations:** The system not only identifies problems but also runs "what-if" simulations to suggest the most effective improvements.
* **Professional Data Visualization:** Automatic generation of a set of 8 detailed and stylish high-resolution charts for a clear presentation of the results.
* **Modular Architecture:** All code is divided into independent modules (`thermal`, `optic`, `indexer`, `recommender`, etc.), ensuring flexibility and scalability.


## Results Gallery

The system automatically generates a full set of visual reports for each analysis.

| Harmony Index | Components | Detailed Metrics | Before & After |
| :---: | :---: | :---: | :---: |
| ![Harmony Gauge](https://github.com/Ilya-arch13/Neighbly.ArchitectureHarmonyIndexSystem/blob/main/visualizer/visualization_results/harmony_gauge_en.png) | ![Subindices Chart](https://github.com/Ilya-arch13/Neighbly.ArchitectureHarmonyIndexSystem/blob/main/visualizer/visualization_results/subindices_chart_en.png) | ![Details View](https://github.com/Ilya-arch13/Neighbly.ArchitectureHarmonyIndexSystem/blob/main/visualizer/visualization_results/details_view_en.png) | ![Before-After Chart](https://github.com/Ilya-arch13/Neighbly.ArchitectureHarmonyIndexSystem/blob/main/visualizer/visualization_results/before_after_chart_en.png) |


## How It Works: Technical Architecture

The project is a data processing pipeline managed by a central script, `pipeline_manager.py`.

1.  **Input Data:** The `pipeline_manager` receives a dictionary with complete room data.
2.  **Analysis:** The data is sequentially passed to the modules:
    * `modules/thermal.py` and `modules/optic.py` for physical calculations.
    * `modules/indexer.py` and `modules/beauty_indexer.py` for index calculations.
    * `modules/recommender.py` for generating recommendations.
3.  **Visualization:** `visualizer/passport.py` receives all calculated data and creates a set of `.png` files.
4.  **Output:** The `pipeline_manager` returns a single, comprehensive dictionary with all analysis results and paths to the created images.


## How to Run and Test

The project is written in Python and can be easily run locally.

**1. Clone the repository:**
```bash
git clone [https://github.com/Ilya-arch13/Neighbly.ArchitectureHarmonyIndexSystem.git](https://github.com/Ilya-arch13/Neighbly.ArchitectureHarmonyIndexSystem.git)
cd Neighbly.ArchitectureHarmonyIndexSystem
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the demonstration:**
```bash
python examples/pipeline_demo.py
```

After running the script, a full analysis report of a sample room will appear in the console, and a set of 8 high-resolution visualizations will be generated in the pipeline_results folder.


## Project Ecosystem

This repository is the "scientific heart" of a larger project. An Android mobile application has already been developed based on this engine, providing users with access to the analysis in a convenient chatbot format. A community is forming around the project in Telegram, where ideas for improving living spaces and the future development of the Index are discussed.


## The Future of the Project

The current system is a powerful tool for analyzing existing spaces. However, this is just the first step. The long-term goal of the project is to use this foundation to create principles for designing adaptive architectural systems. These are spaces that can not only be statically "good" but can also independently analyze their state and dynamically change to maintain the highest level of harmony, responding to changes in the environment and human needs.
