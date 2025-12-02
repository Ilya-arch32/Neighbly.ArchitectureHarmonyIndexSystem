## Architectural Harmony Index System

**AHI** is a multiphysics digital twin platform for assessing **"Architectural Harmony."** The system combines **engineering simulation** (*CFD, acoustics, energy efficiency*) with **cognitive neuroscience metrics** (*neuroaesthetics*) to create environments optimized for human health and perception.

**Current status** (*TRL 6*): The system has been demonstrated as a fully functional prototype in a relevant environment. The physics cores (LBM, FDTD) have been internally validated on benchmarks. The project is being prepared for pilot testing with specialized experts.

# Key Features
**1. Multiphysics Core (WebGPU)**

The system performs calculations directly in the browser, leveraging the computing power of the GPU:

- **Aerodynamics** (LBM-LES): Lattice Boltzmann method (D3Q19) with the Smagorinsky turbulence model. Simulates natural ventilation and thermal convection.

- **Thermodynamics** (CHT + ISO 13788): Conjugate heat transfer with calculation of mold risk (moisture transport) and material buffering capacity (MBV).

- **Acoustics** (FDTD): Finite difference time-domain method. Calculates RT60, C50, and speech intelligibility using Schroeder integration, taking diffraction into account.

**2. Neuroaesthetic Analysis**

For the first time in the BIM industry, aesthetics are quantified:

- **Fractal Dimension** (3D Box-Counting): Assessing the biophilic potential of architecture (D≈1.3−1.5).

- **Visual Entropy**: Scene complexity analysis to predict cognitive load.

- **Spectral Tracing**: Calculating circadian rhythms (EML/CS) taking into account the spectral properties of materials, not just luminance.

**3. Autonomous Optimization**

**NSGA-III**: A genetic algorithm optimizes room geometry based on 10+ conflicting criteria (e.g., "Maximum Heat" vs. "Minimum Glare").

# Installation and Launch

**Requirements** 

- **Browser with WebGPU support** (Chrome 113+, Edge).

- **Python 3.10+** (for the weather generation backend).

- **Node.js 18+** (for building the frontend).

**Build bash**

1. Clone the repository
```
git clone https://github.com/your-username/ahi-ultimate.git
```
2. Install frontend dependencies
```
cd ahi-ultimate npm install
```
3. Run the backend (in a separate terminal)
```
cd backend pip install -r requirements.txt python main_analysis_server.py
```
4. Run the client
```
npm run dev
```
## Validation and TRL
The system includes the automatic self-diagnostic module `PhysicsBenchmarks.ts`. Upon startup, the following are checked:

- **Poiseuille Flow:** L2 Error < 1% (passed).

- **Lid-Driven Cavity:** Re=1000, eddy matching with Ghia et al. data (passed).

- **Acoustics:** Convergence of the FDTD method with Sabine diffuse field theory.

## Collaboration
We invite **architects**, **HVAC engineers**, and **neuroscience researchers** to participate in beta testing (progressing to TRL 7).
Please open Issues for suggestions for improving the physics models or UX.

## License
This project is licensed under the **GNU AGPL v3.0** license.
*If you use this code in a web service, **you must make your modifications open source***.

## Contact Me

* **Telegram** - @Shelookslikefunn
* **LinkedIn** - [tap!](https://www.linkedin.com/in/ilya-trofimov-64a779380/)

Also check out my **Neighbly** application, which'll get a **massive update soon** (**will include this Index**)!

* **Google Play** - [tap!](https://play.google.com/store/apps/details?id=com.neighbly.app)
* **Telegram Community** - [tap!](t.me/Neighbly)

Additional links (for your convenience)

* **Instagram** - @ilyatrofimov_architecture
* **X (Twitter)** - @ilytrof_arch
* **Reddit** - [tap!](https://www.reddit.com/u/legsnfeetss/s/5gJj6djRKh)
* **Gmail** - Sarkovvitala@gmail.com
