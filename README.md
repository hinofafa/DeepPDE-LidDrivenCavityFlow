# Lid Driven Cavity Flow

The lid-driven cavity is a popular problem within the field of computational fluid dynamics (CFD) for validating computational methods. In this repository, we will walk through the process of generating a 2D flow simulation for the Lid Driven Cavity (LDC) flow using Nvidia Modulus framework.

## Problem Description

The lid-driven cavity is a well-known benchmark problem for viscous incompressible fluid flow [[75]](https://www.sciencedirect.com/book/9781856176354/the-finite-element-method-for-fluid-dynamics). The geometry at stake is shown in Figure 27. We are dealing with a square cavity consisting of three rigid walls with no-slip conditions and a lid moving with a tangential x-direction velocity of 1m/s. The [Reynolds number](https://en.wikipedia.org/wiki/Reynolds_number) based on the cavity height is chosen be 10.

<img src="pictures/lidrivencavity.PNG"  align='center'/>


## Steps (Modulus)
1. - Geometry setup (Constructive Solid Geometry, CSG or STL)
   - Sampling on boundaries and interior regions
2. Equation setup (TrainDomain)
3. Solver setup (Solver)

<img src="pictures/Methodology.PNG"  align='center'/>

     

 