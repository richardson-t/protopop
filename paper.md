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

Alternate approaches have been developed to produce protocluster models. @myers2011, @myers2012, @myers2014 and @zhou2024 create analytic or semianalytic procedures to translate parameterizations of star formation theory (star formation rates, accretion histories, PMS stellar evolutionary tracks, etc.) into protostellar populations, similar to `protopop`. However, approaches of this kind generally do not build in radiative transfer, so the predictions from the resulting models are limited to summary properties--chiefly the masses and bolometric luminosities of protostars--instead of direct observables. Some models involving radiative transfer output also exist; the protocluster model grid of @molinari2019 makes predictions in specific wavelength bands by compositing existing radiative transfer models, similar to `protopop`. This grid, however, has few other analogues and does not create fully self-consistent cluster models with evolutionary histories. Furthermore, it is limited to only four tunable parameters which do not directly relate to star formation theory, making it difficult to use as a test of theory. @peltonen2026 (including the core developer of `protopop`) recently developed a synthesis approach for cloud-scale protostellar populations with a greater amount of self-consistency and theoretical flexibility, but it is developed primarily to model large-scale and unresolved populations and predict radiation in the infrared for comparison to extragalactic data rather than portray the myriad wavelengths available for nearby protoclusters.

Taking a wider view of current protocluster modeling approaches, they are generally not implemented as open-source software, meaning that there are few available ways for the general community to replicate these approaches or edit the underlying theoretical assumptions to model different scenarios. There have been efforts to use stellar population synthesis tools which are open source and permit more theoretical flexibility as conduits to model protoclusters [e.g. @pedrini2025], but these generally lack the ability to accurately model pre- and protostellar evolution and the corresponding radiation profiles. `protopop` is built to address this gap in modeling capability. It translates star formation theories to predictions in the same way as existing semianalytic procedures and is based on self-consistent protostellar evolutionary histories connected to full radiative transfer, providing an informed picture of radiation covering a wide swathe of the spectrum.

# Software design

Research applications of population synthesis tools (find some galaxy-scale/IMF stuff?) necessitate the ability to compare synthesized populations to data. They also typically benefit from considering multiple physical scenarios, as expanding the theory space that can be translated into predictions allows a fuller understanding of which theories are consistent or inconsistent with observational results. Consequently, an expected workflow for a project making use of this package includes the creation of multiple populations, which requires performing the same operations across a potentially very wide theory space. protopop supports such a workflow through an object-oriented design, where each population is contained in a saveable "Cluster" object with a suite of tunable theoretical attributes and class methods allowing access to the physical properties and radiation emitted by individual population members at any given time. This design is intended to make the core functions of generating and sampling protocluster models easy to understand and execute while allowing generated models to persist and preserve their information.

%why interpolation?
-tracking member evolution is important to be able to match the full scope of available observations
  -each member has an evolutionary history in physical properties and flux (a lot of information given that SEDs are defined at 200 wavelengths)
-clusters can reach thousands+ of members
-protopop interpolates to avoid associating a set track with each member, which would be very memory intensive given the size of template files (check how much an evolutionary track + a flux track would be per member)

# Research impact statement

%cite/allude to research using this tool; should include your work, possibly Taehwa's work, seshat 2.0, etc.

`protopop` is a central pillar of the core developer's present work, where it has been used to demonstrate the differentiability of star formation theories on a cluster scale by comparing the flux distributions sampled from models generated with the code [@richardsonfuture]; the accompanying set of models is publicly available alongside the package. The population models and flux distributions generated with `protopop` are designed to be a general point of comparison for observations; this potential has been realized in an analysis of multi-wavelength data taken in Galactic star-forming regions (Taehwa). This package is also a primary source of training data for `seshat`, a Python-based tool for classifying objects [@crompvoets2026], which is a central component of analyzing data from astronomical surveys. An iteration of `seshat` which relies on model populations created with `protopop` to establish the infrared colors likely to be exhibited by forming stars is currently in active development.
%if dataset public, use that as "benchmark/reproducible materials"

# AI usage disclosure

No generative AI tools were utilized in the process of creating and documenting this software package or in the process of paper authoring.

# Acknowledgements

This work was supported by NSF grants AST 2008101 and CAREER 2142300 (PI: Adam Ginsburg). I acknowledge [UFIT Research Computing](https://it.ufl.edu/rc) for providing computation resources and support that have contributed to the research results reported in this publication. The package makes use of the following software: astropy [@astropy2013; @astropy2018; @astropy2022], astroquery [@astroquery], [h5py](https://www.h5py.org/), imf [@imf], numpy [@numpy], scipy [@scipy].

# References
