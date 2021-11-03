
import kfp
from kfp.v2 import compiler
from kfp.v2.google.client import AIPlatformClient
from google.cloud import aiplatform
from google.cloud import aiplatform as aip
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.v2 import compiler
from datetime import datetime

# Change the Project ID
#PROJECT_ID ="[YOUR PROJECT ID]"
#BUCKET_LOC = "[Your staging bucket locations]"
#PIPELINE_NAME="[Your Pipeline Name]"

PROJECT_ID ="demogct"
BUCKET_LOC = "gs://demogct/vipipelines/"
PIPELINE_NAME="sd-vertex-pipeline"

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
DISPLAY_NAME = "flowers_" + TIMESTAMP
PIPELINE_ROOT="gs://demogct/vipipelines/"

job = aip.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path="artifacts/image classification_pipeline.json".replace(" ", "_"),
    pipeline_root=PIPELINE_ROOT,
)

job.run()
