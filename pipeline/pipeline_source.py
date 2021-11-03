#This is a simple pipeline that is used to load and run pipeline

import kfp
from kfp.v2 import compiler
from kfp.v2.google.client import AIPlatformClient
from google.cloud import aiplatform
from google.cloud import aiplatform as aip
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.v2 import compiler

# Change the Project ID
#PROJECT_ID ="[YOUR PROJECT ID]"
#BUCKET_LOC = "[Your staging bucket locations]"
#PIPELINE_NAME="[Your Pipeline Name]"

PROJECT_ID ="demogct"
BUCKET_LOC = "gs://demogct/vipipelines/"
PIPELINE_NAME="sd-vertex-pipeline"


aip.init(project=PROJECT_ID, staging_bucket=BUCKET_LOC)

@kfp.dsl.pipeline(name=PIPELINE_NAME)
def pipeline(project: str = PROJECT_ID):
    ds_op = gcc_aip.ImageDatasetCreateOp(
        project=project,
        display_name="flowers",
        gcs_source="gs://cloud-samples-data/vision/automl_classification/flowers/all_data_v2.csv",
        import_schema_uri=aip.schema.dataset.ioformat.image.single_label_classification,
    )

    training_job_run_op = gcc_aip.AutoMLImageTrainingJobRunOp(
        project=project,
        display_name="train-automl-flowers",
        prediction_type="classification",
        model_type="CLOUD",
        base_model=None,
        dataset=ds_op.outputs["dataset"],
        model_display_name="train-automl-flowers",
        training_fraction_split=0.7,
        validation_fraction_split=0.2,
        test_fraction_split=0.1,
        budget_milli_node_hours=8000,
    )
    # 0.1.7 is needed currently to address a bug in the latest container image
    gcc_aip.ModelDeployOp.component_spec.implementation.container.image = ("gcr.io/ml-pipeline/google-cloud-pipeline-components:0.1.7")
 
    
    endpoint_op = gcc_aip.ModelDeployOp(  
         model=training_job_run_op.outputs["model"],
         project="demogct"
    )
  


compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path="artifacts/image classification_pipeline.json".replace(" ", "_"),
)
