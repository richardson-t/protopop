title: 'protopop: Stellar protocluster synthesis in Python'
tags:
  - Python
  - astronomy
  - star formation
  - protoclusters
authors:
  - name: Theo Richardson
    orcid: 0009-0001-8880-6951
    corresponding: true
    affiliation: 1
affiliations:
 - name: University of Florida, United States
   index: 1
date: 24 October 2026
bibliography: paper.bib

# Summary

text

# Statement of need

-Self-consistent stellar populations are a standard entry in the toolkit for research on stars and galaxies because they present a physically motivated point of comparison
-Modern surveys of Galactic/extragalactic star formation (HiGAL, ALMA-IMF, ALMAGAL) provide enough data to merit the same kind of population-level analysis
-However, no tools exist which create self-consistent models or allow the same level of theoretical flexibility as typical SPS

# State of the field

-SPS: typically doesn't extend into pre-/protostellar phases
-Similar PMS cluster modeling approaches:
    -Analytic and semianalytic models (Myers/Zhou) typically focus on summary properties, don't do radiative transfer, build in particular theories
    -Sergio Molinari's cluster models (no self-consistent histories or persistent members, theory not customizable)
  -broader point: no existing approaches really allow for wider generation of new models or much customization, because they're not implemented as open access software

# Software design

%why OOP?
Research applications of population synthesis tools (find some galaxy-scale/IMF stuff?) necessitate the ability to compare synthesized populations to data. They also typically benefit from considering multiple physical scenarios, as expanding the theory space that can be translated into predictions allows a fuller understanding of which theories are consistent or inconsistent with observational results. Consequently, an expected workflow for a project making use of this package includes the creation of multiple populations, which requires performing the same operations across a potentially very wide theory space. protopop supports such a workflow through an object-oriented design, where each population is contained in a saveable "Cluster" object with a suite of tunable theoretical attributes and class methods allowing access to the physical properties and radiation emitted by individual population members at any given time. This design is intended to make the core functions of generating and sampling protocluster models easy to understand and execute while allowing generated models to persist and preserve their information.

%why interpolation?
-tracking member evolution is important to be able to match the full scope of available observations
  -each member has an evolutionary history in physical properties and flux (a lot of information given that SEDs are defined at 200 wavelengths)
-clusters can reach thousands+ of members
-protopop interpolates to avoid associating a set track with each member, which would be very memory intensive given the size of template files (check how much an evolutionary track + a flux track would be per member)

# Research impact statement

text. (cite/allude to research using this tool; should include your work, possibly Taehwa's work, seshat 2.0, etc.)

test citation: @Pearson:2017

# AI usage disclosure

No generative AI tools were utilized in the process of creating and documenting this software package or in the process of paper authoring.

# Acknowledgements

This work was supported by NSF grants AST 2008101 and CAREER 2142300 (PI: Adam Ginsburg). I acknowledge [UFIT Research Computing](https://it.ufl.edu/rc) for providing computation resources and support that have contributed to the research results reported in this publication. This package makes use of the following software: astropy [@astropy2013; @astropy2018; @astropy2022], astroquery [@astroquery], [h5py](https://www.h5py.org/), imf [@imf], numpy [@numpy], scipy [@scipy].

# References
