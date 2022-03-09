## Kubeflow Pipelines

A pipeline is a description of an ML workflow, including all of the components in the workflow and how they combine in the form of a graph. (See the screenshot below showing an example of a pipeline graph.) 

The pipeline includes the definition of the inputs (parameters) required to run the pipeline and the inputs and outputs of each component.

A pipeline component is a self-contained set of user code, packaged as a Docker image, that performs one step in the pipeline.
E.g. a component can be responsible for data preprocessing, data transformation, model training, and so on.

Kubeflow Pipelines is a platform for building and deploying portable, scalable machine learning (ML) workflows based on Docker containers.

## Kubeflow Components

### Pipeline
A pipeline is a description of a machine learning (ML) workflow, including all of the components in the workflow and how the components relate to each other in the form of a graph. The pipeline configuration includes the definition of the inputs (parameters) required to run the pipeline and the inputs and outputs of each component.

When you run a pipeline, the system launches one or more Kubernetes Pods corresponding to the steps (components) in your workflow (pipeline). The Pods start Docker containers, and the containers in turn start your programs.
After developing your pipeline, you can upload your pipeline 

### Component
A pipeline component is a self-contained set of code that performs one step in the ML workflow (pipeline), such as data preprocessing, data transformation, model training, and so on. A component is analogous to a function, in that it has a name, parameters, return values, and a body.
Component definition
A component specification in YAML format describes the component for the Kubeflow Pipelines system. A component definition has the following parts:
Metadata: name, description, etc.
Interface: input/output specifications (name, type, description, default value, etc).
Implementation: A specification of how to run the component given a set of argument values for the component’s inputs. The implementation section also describes how to get the output values from the component once the component has finished running.
For the complete definition of a component, see the component specification.

### Containerizing components
You must package your component as a Docker image. Components represent a specific program or entry point inside a container.
Each component in a pipeline executes independently. The components do not run in the same process and cannot directly share in-memory data. You must serialize (to strings or files) all the data pieces that you pass between the components so that the data can travel over the distributed network. You must then deserialize the data for use in the downstream component.

### Graph
A graph is a pictorial representation in the Kubeflow Pipelines UI of the runtime execution of a pipeline. The graph shows the steps that a pipeline run has executed or is executing, with arrows indicating the parent/child relationships between the pipeline components represented by each step. The graph is viewable as soon as the run begins. Each node within the graph corresponds to a step within the pipeline and is labeled accordingly.

 
### Step
Conceptual overview of steps in Kubeflow Pipelines
A step is an execution of one of the components in the pipeline. The relationship between a step and its component is one of instantiation, much like the relationship between a run and its pipeline. In a complex pipeline, components can execute multiple times in loops, or conditionally after resolving an if/else like clause in the pipeline code.

### Output Artifact
Conceptual overview of output artifacts in Kubeflow Pipelines
An output artifact is an output emitted by a pipeline component, which the Kubeflow Pipelines UI understands and can render as rich visualizations. It’s useful for pipeline components to include artifacts so that you can provide for performance evaluation, quick decision making for the run, or comparison across different runs. Artifacts also make it possible to understand how the pipeline’s various components work. An artifact can range from a plain textual view of the data to rich interactive visualizations.

### ML Metdata

ML Metadata (MLMD) is a library for recording and retrieving metadata associated with ML developer and data scientist workflows. 
MLMD helps you understand and analyze all the interconnected parts of your ML pipeline instead of analyzing them in isolation and can help you answer questions about your ML pipeline such as:
* Which dataset did the model train on?
* What were the hyperparameters used to train the model?
* Which pipeline run created the model?
* Which training run led to this model?
* Which version of TensorFlow created this model?
* When was the failed model pushed?

### Metadata store
MLMD registers the following types of metadata in a database called the Metadata Store.
Metadata about the artifacts generated through the components/steps of your ML pipelines
Metadata about the executions of these components/steps
Metadata about pipelines and associated lineage information
The Metadata Store provides APIs to record and retrieve metadata to and from the storage backend. 
