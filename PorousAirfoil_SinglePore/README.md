# PorousAirfoil_PotentialFlow

This repository implements a **Source Panel Method with Vortex correction (SPVP)** to simulate airflow around NACA airfoils and compute pressure distributions, lift coefficients (CL), and moment coefficients (CM). The tool also compares results against reference data generated using **XFOIL**.

---

## 🗂️ Repository Structure

- `AoA_Sweep.py`  
  This script compares the aerodynamic performance of a porous and a solid airfoil using the SPVP method across a range of angles of attack. It computes and plots the lift (CL), drag (CD), and lift-to-drag ratio (CL/CD) for both configurations.

- `SPVP_Airfoil.py`  
  This script implements the Source and Vortex Panel Method (SPVP) to compute the pressure distribution, lift, moment, and drag coefficients of a NACA 4-digit airfoil. It supports both porous and solid configurations and provides detailed visualizations of aerodynamic quantities.

- `PLOT.py`  
    This module provides a suite of visualization tools for the SPVP method, including plots for airfoil geometry, normal vectors, pressure coefficients, streamlines, and pressure contours.

- `COMPUTATION/`  
  - `COMPUTE.py`: It includes functions to build influence matrices, solve the linear system, and compute lift, drag, and moment coefficients based on airfoil geometry and flow characteristics.
  - `Hydraulic_Resistance.py`: Models resistances for porous airfoil simulation.

- `GEOMETRY/`  
  - `GEOMETRY.py`: Constructs panel geometry for the airfoil.  
  - `Hydraulic_GEOMETRY.py`: Adapts geometry for porous surfaces.  
  - `NACA.py`: Generates 4-digit NACA airfoil coordinates.  
  - `PANEL_DIRECTION.py`: Changes the panels directions if needed.

- `X_FOIL/`  
  - `X_FOIL.py`: Reads XFOIL-generated data files.  
  - `X_FOIL_DATA/`: Directory containing `.dat` files exported from XFOIL.

- `Example1.py`
  This example demonstrates the SPVP panel method to simulate a solid airfoil’s aerodynamics by defining its geometry, calculating aerodynamic coefficients, and visualizing pressure distribution and streamlines.

- `Example2.py`
  This example applies the panel method to porous airfoils, computing aerodynamic coefficients and visualizing differences with solid airfoils.

---

## ⚙️ Features

- **SPVP Solver**  
  Models inviscid, incompressible flow using source and vortex panels.

- **Lift and Moment Computation**  
  Integrates pressure distribution to compute CL and CM.

- **XFOIL Integration**  
  Imports `.dat` files generated with XFOIL for validation and comparison.

- **Porous Surface Modeling**  
  Adds porous resistance through hydraulic modeling components.

---

## 🚀 Getting Started

### 1. Requirements

Install required dependencies:

```bash
pip install numpy 
pip install matplotlib
pip install scipy
```
### 2. Examples

#### Example 1 : Solid Airfoil
This example demonstrates the use of the panel method (SPVP) to simulate the aerodynamic behavior of a solid airfoil.  
It shows how to define the airfoil geometry, compute aerodynamic coefficients, and visualize results such as pressure distribution and streamlines.

#### Example 2 : Porous Airfoil

This example demonstrates the application of the panel method adapted for porous airfoils. It computes the aerodynamic coefficients and pressure distributions for a NACA 4-digit airfoil with porous panels, accounting for hydraulic resistance and pore geometry effects. Various plots visualize the flow characteristics and compare porous versus solid airfoil results.