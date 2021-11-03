## About

This workshop shows how to do a CI/CD with MLOps. This is a continuation of the Day 3 workshop where we will take the IRIS Dataset and use automl classificaton to do a classification on images of flowers and then deploy the model. In this demo we will show how to use a) Cloud Source Repoitory, b) Cloud Build and trigger automated and manual builds. 

### Enviroment
This package is assumed to be be executed in vertex ai notebooks environment.
To perform the CI CD, we will use Vertex AI Piple code in Kubeflow.
 
### Code Organization
The code structure is defined in the following folders:

- notebook:
    This contains the notebooks or experiments that you are working with. This has 3 files
      - SourceRepoSetup.ipynb - This file lets you setup a code repo in Google Cloud Source Repository, create a cloud build and execute
      - IrisflowersAutoMLKubeflowPipeline.ipynb - This notebook shows the pipeline that can be executed cell by cell, to understand the pipeline flow.
      - IrisPipelineTemplate.ipynb - This notebook generates two pipeline files that can be used to by the build system
- pipeline:
    This folder containers the trainer code pipeline that is for model training
- artifacts"
    This is the docker file and other artifacts. This is optional and can be used if you want just have a training image that you would want to build out.

Following files in the root of the folder:
- cloudbuild.yaml:
     This is the build file used by cloud build. This has two steps one for build and one for execution of the pipeline.
- requirements.txt:
     Python packages needed to perform the build

Steps to execute for this

Step 1:
- Setup the source repository.
   Follow the instrustions in the "SourceRepoSetup.ipynb" to setup a source repository.
   Make sure you clone the code the location where you have unzipped this code. The root of the folder should be the home of the repo.
   You can check on the Cloud Source UI the repo you have created. 
   Add the files to the source repository
Setup Cloud Build
- Go to the Cloud Build and setup a Trigger. Follow the default wizard on the page
   - Select the Source Repo
   - Select the default cloudbuild.yaml file and hit save.
 
Step2:
- Explore the pipeline code (IrisflowersAutoMLKubeflowPipeline.ipynb)
   You may have done this in the Day3 workshop. We are going to work with the iris dataset to classiify flower images. This is fairly simple where you will use the dataset creation and AutoML to classify the images. At the end you will deploy the model.
   
Step3:
- Prepare the pipeline python code. Execute the notebook "IrisPipelineTemplate.ipynb". Change the needed variables in the code and generate the pipeline files. The pipeline files should be generated in the pipeline folder. There are two files. One is for the pipeline source and the other takes the compiled pipeline output and executes it.

Step 4:
- Manually Execute the cloud build

Step 5:
- Change the name of the pipeline in the "IrisPipelineTemplate.ipynb" file. Execute the cells to generate the files.

Step 6:
 - Add files to the git repository 
    - git add .
    - git commint -m "some message"
    - git push 
  
  Now open the cloud build console UI. You should see a build kicked off.
  You can navigate to Vertex AI Pipelines, you will see a pipeline launched.


   



