---
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
 - name: Department of Astronomy, University of Florida, PO Box 112055, Gainesville, FL, USA
   index: 1
date: 3 November 2026
bibliography: paper.bib
---

# Summary

Star formation is a key part of the evolution of matter in the universe, shaping the future development of structures from planetary to galactic scales. However, uncertainties remain in the picture of how stars grow and how the properties of newly formed stellar populations are set, and consequently the physics governing the process of star formation remain somewhat unconstrained. Efforts to study later stages of the stellar life cycle have been greatly buoyed by the technique of population synthesis, which creates detailed, physically motivated, and internally consistent models of stellar populations. These models allow straightforward comparison between model and data and therefore permit the theory underlying the model to be tested. `protopop` is a software package written in Python which provides a simple and flexible tool for synthesizing populations of pre- and protostellar objects, enabling more substantive research in the earliest phases of star formation.

# Statement of need

Modern surveys of star formation in the Milky Way [@sanhueza2019; @motte2022; @molinari2025] have provided a wealth of observational evidence for how local star formation proceeds. Data on forming stellar populations--hereinafter "protoclusters"--is now available across the Galactic plane spanning multiple stages of protocluster evolution. However, there are few widely available model sets or modeling tools that can be used in analysis of this data, meaning that the ability to derive theoretical significance from the observations is limited. Extragalactic studies of star formation have been greatly advanced through the existence of population synthesis techniques for star clusters [@conroy2013], which are now a standard component in most analyses. There are many pulicly available and heavily used software tools for modeling star clusters based on main-sequence and post-main-sequence stellar evolution [e.g. @bruzual2003; @conroy2009; @krumholz2015] which produce testable predictions comparable to observed simple and composite stellar populations.

`protopop` is a software package for detailed and theoretically flexible protocluster modeling, enabling more throrough analysis of protocluster data. It is implemented in Python for easy integration with the wider ecosystem of Pythonic astronomy software. The API is designed to be simple and easily comprehensible for researchers in astronomy across fields and experience levels.

# State of the field

Alternate approaches have been developed to produce protocluster models. Analytic or semianalytic procedures have been created to translate parameterizations of star formation theory (star formation rates, accretion histories, PMS stellar evolutionary tracks, etc.) into protostellar populations, similar to `protopop` [@myers2014; zhou2024]. However, approaches of this kind generally do not build in radiative transfer, so the predictions from the resulting models are limited to summary properties--chiefly the masses and bolometric luminosities of protostars--instead of direct observables. Some models involving radiative transfer output also exist; the protocluster model grid of @molinari2019 makes predictions in specific wavelength bands by compositing existing radiative transfer models, similar to `protopop`. This grid, however, has few other analogues and does not create fully self-consistent cluster models with evolutionary histories. Furthermore, it is limited to only four tunable parameters which do not directly relate to star formation theory, making it difficult to use as a test of theory. @peltonen2026--including the core developer of `protopop`--recently developed a synthesis approach for cloud-scale protostellar populations with a greater amount of self-consistency and theoretical flexibility, but it is developed primarily to model large-scale and unresolved populations and predict radiation in the infrared for comparison to extragalactic data rather than portray the myriad wavelengths available for nearby protoclusters.

Taking a wider view of current protocluster modeling approaches, they are generally not implemented as open-source software, meaning that there are few available ways for the community to replicate these approaches or edit the underlying theoretical assumptions to model different scenarios short of entirely recreating the described infrastructure. There have been efforts to use stellar population synthesis tools which are open source and permit more theoretical flexibility as conduits to model protoclusters [e.g. @pedrini2025], but these generally lack the ability to accurately model pre- and protostellar evolution and the corresponding radiation profiles. `protopop` is built to address this gap in modeling capability. It translates star formation theories to predictions in the same way as existing semianalytic procedures and is based on self-consistent protostellar evolutionary histories connected to full radiative transfer, providing an informed picture of radiation covering a wide swathe of the spectrum.

# Software design

Research applications of population synthesis tools usually require comparing synthesized populations to data. They also typically benefit from considering multiple physical scenarios, as expanding the theory space that can be translated into predictions allows a fuller understanding of which theories are consistent or inconsistent with observational results. Consequently, an expected workflow for a project making use of this package includes the creation of multiple populations, which requires performing the same operations across a potentially very wide theory space. protopop supports such a workflow through an object-oriented design, where each population is contained in a saveable "Cluster" object with a suite of tunable theoretical attributes and class methods allowing access to the physical properties and radiation emitted by individual population members at any given time. This design is intended to make the core functions of generating and sampling protocluster models easy to understand and execute while allowing generated models to persist and preserve their information.

`protopop` predicts the properties and radiation emitted by cluster members by interpolating between precomputed tables tracking the evolution of protostars and corresponding fluxes. The objective of the package is to make models which are self-consistent all the way down to individual protocluster members. Producing these self-consistent predictions requires input from a protostellar evolutionary code [e.g. @hosokawa2009, @klassen2012] in order to track the growth rates of protostars, which affect the flow of material and brightness of members within the protocluster; further, the output from this code must be extended via the algorithm of @richardson2025 in order to produce flux predictions meeting the package's wider objectives. These steps must be involved either at the time of creating a new cluster member or by precomputing and interpolating. `protopop` adopts the latter in recognition of the cost of running and storing unique tracks for each member; calculating an evolutionary and flux history typcially takes on the order of 60+ seconds per protostar, and a full flux track spanning the wavelength range accessible by the base radiative transfer models (necessary to take full advantage of modern data) has a typical size of approximately 200 MB. For a cluster of 1000 stars, creating and sampling a population by generating unique output could take upwards of half a day and occupy several hundred GB. By comparison, a predefined set of lookup tables limits the additional necessary memory usage beyond approximately 50GB and greatly reduces the time required to sample at the cost of slightly lower accuracy in the predicted values resulting from interpolation. Structuring the code to reference lookup tables also allows for straightforward expansion of modeling capabilities beyond those present in the current release by simply adding tables that incorporate new models.

# Research impact statement

`protopop` is a central pillar of the core developer's present work, where it has been used to demonstrate the differentiability of star formation theories on a cluster scale by comparing the flux distributions sampled from models generated with the code [@richardsonfuture]; the accompanying set of models is publicly available alongside the package. The population models and flux distributions generated with `protopop` are designed to be a general point of comparison for observations; this potential has been realized in an analysis of multi-wavelength data taken in Galactic star-forming regions (Taehwa). This package is also a primary source of training data for `seshat`, a Python-based tool for classifying objects [@crompvoets2026], which is a central component of analyzing data from astronomical surveys. An iteration of `seshat` which relies on model populations created with `protopop` to establish the infrared colors likely to be exhibited by forming stars is currently in active development.

# AI usage disclosure

No generative AI tools were utilized in the process of creating and documenting this software package or in the process of paper authoring.

# Acknowledgements

This work was supported by NSF grants AST 2008101 and CAREER 2142300 (PI: Adam Ginsburg). I acknowledge [UFIT Research Computing](https://it.ufl.edu/rc) for providing computation resources and support that have contributed to the research results reported in this publication. The package makes use of the following software: astropy [@astropy2013; @astropy2018; @astropy2022], astroquery [@astroquery], [h5py](https://www.h5py.org/), imf [@imf], numpy [@numpy], scipy [@scipy].

# References
