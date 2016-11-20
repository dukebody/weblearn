# WebLearn

This is a proof of concept to create a very simple component to serve
scikit-learn machine learning models over HTTP. The idea is to use this
component as a micro-service inside a more complex infrastructure.

It aims to provide a set of base classes, glue code and code patterns to make
the deployment of these models easy and writing very few lines of code.

It's based on Flask because it's lightweight, easy to use and fairly well
documented.

See an example of use in the `iris` module.

# Features roadmap

 - Input validation
 - Input transformations
