
&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;<img src="images/vertexai.png" width="200" height="200"/> &nbsp; &nbsp; &nbsp; &nbsp; <img src="images/kubeflow.png" width="400" height="200"/>&nbsp; &nbsp;

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Pipelines

Before we get started it is a good pre-read to understand the concepts of Kubleflow. The [Kubeflow.md](Kubeflow.md) file covers the basics and understanding on Kubeflow. After finishing the Kubelflow.md, [Vertex Pipeline User Guide](Vertex Pipeline User Guide.pdf) provides a good introduction to Vertex AI Pipelines and how it is different from the Open Source Kubeflow pipelines.


## About

I this workshop you will do the following
* Setup your lab environment
* Run a basic intro to kubeflow lab
* Run a Kubelfow controls
* Build a sample Machine Learning Pipeline
* Trigger the machine learning pipeline using Cloud Functions
* Setup Source Control with Cloud Source Repository
* Setup Build Trigger
* Run CI/CD with MLOps Pipelines

## Qwiklabs Pipelines 
If you are interested in doing the Qwiklabs, it has a good step by step process
![Qwiklabs Tutorials](https://www.cloudskillsboost.google/focuses/21234?parent=catalog)

### Enviroment

Setup the environment with following [pre steps](pre-steps.md)

 
### Code Organization
The code structure is defined in the following folders:

- **notebook**:
    This contains the notebooks for KFP and CI/CD.
    
    * **[01pipelines_intro_kfp.ipynb](notebooks/01pipelines_intro_kfp.ipynb)** 
                  This is an intro notebook to Pipelines
    * **[02control_flow_kfp.ipynb](notebooks/02control_flow_kfp.ipynb)** (Optional)
                  This a second KFP Pipelines showing how to work with control flows and parallel execution
    * **[03IrisflowersAutoMLKubeflowPipeline.ipynb](notebooks/03IrisflowersAutoMLKubeflowPipeline.ipynb)**:
                  This notebook shows the pipeline that can be executed cell by cell, to understand the pipeline flow.
    * **[04SourceRepoSetup.ipynb](notebooks/04SourceRepoSetup.ipynb)**: `IGNORE THIS FILE `
                  This file lets you setup a code repo in Google Cloud Source Repositorycreate a cloud build and execute
    * **[05IrisPipelineTemplate.ipynb](notebooks/05IrisPipelineTemplate.ipynb)**: 
                  This notebook generates two pipeline files that can be used to by the build system
      
- **pipeline**:
    This folder containers the trainer code pipeline that is for model training
- **artifacts**:
    This is the docker file and other artifacts. This is optional and can be used if you want just have a training image that you would want to build out.

Following files in the root of the folder:
- **cloudbuild.yaml**:
     This is the build file used by cloud build. This has two steps one for build and one for execution of the pipeline.
- **requirements.txt**:
     Python packages needed to perform the build

### Steps to execute for this

Complete the [pre_steps](pre_steps.md) if you have not
* **Step1**
  Complete `01pipelines_intro_kfp.ipynb`

* **Step2**
  Complete **[02control_flow_kfp.ipynb](notebooks/02control_flow_kfp.ipynb)** 
  
* **Step2**:
- Explore the pipeline code (IrisflowersAutoMLKubeflowPipeline.ipynb)
   We are going to work with the iris dataset to classiify flower images. This is fairly simple where you will use the dataset creation and AutoML to classify the images. At the end you will deploy the model.
   
**Step3: Running Cloud Build Trigger and Pipeline**:
- Prepare the pipeline python code. Execute the notebook "IrisPipelineTemplate.ipynb". Change the needed variables in the code and generate the pipeline files. The pipeline files should be generated in the pipeline folder. There are two files. One is for the pipeline source and the other takes the compiled pipeline output and executes it.

**Step 4**:
- Manually Execute the cloud build
* `gcloud builds submit --config cloudbuild.yaml --timeout=1000`

**Step 5**:
- Change the pipeline parameters of the pipeline in the "IrisPipelineTemplate.ipynb" file. Execute the cells to generate the files.

**Step 6**:
 - Add files to the git repository 
    - git add .
    - git commit -m "some message"
    - git push 
  
  Now open the cloud build console UI. You should see a build kicked off.
  You can navigate to Vertex AI Pipelines, you will see a pipeline launched.


   



