## [Google Colab](https://colab.corp.google.com/)

Google Colab is a popular and free to use jupyter notebooks hosted by Google Research. Colabs can runtimes can be
- Hosted VM
- GCE Runtime (GCP Marketplace Colab instance)
- Local Runtime

The GCE Runtime is flaky and does not run well. The other option is to host it with GCE instance.


### Connecting to a runtime on a Google Compute Engine instance

If the Jupyter notebook server you'd like to connect to is running on another machine (e.g. Google Compute Engine instance), you can set up SSH local port forwarding to allow Colaboratory to connect to it.

Note: Google Cloud Platform provides Deep Learning VM images with Colaboratory local backend support preconfigured. Follow the how-to guides to set up your Google Compute Engine instance with local SSH port forwarding. If you use these images, skip directly to Step 4: Connect to the local runtime (using port 8888).

First, set up your Jupyter notebook server using the following instructions:
    pip install --upgrade jupyter_http_over_ws>=0.0.7 &&   jupyter serverextension enable --py jupyter_http_over_ws
    jupyter notebook   --NotebookApp.allow_origin='https://colab.research.google.com'   --port=8888   --NotebookApp.port_retries=0

Second, establish an SSH connection from your local machine to the remote instance (e.g. Google Compute Engine instance) and specify the '-L' flag. For example, to forward port 8888 on your local machine to port 8888 on your Google Compute Engine instance, run the following:

gcloud compute ssh --zone YOUR_ZONE YOUR_INSTANCE_NAME -- -L 8888:localhost:8888
    
#### Connect to the local runtime
In Colaboratory, click the "Connect" button and select "Connect to local runtime...". Enter the URL from the previous step in the dialog that appears and click the "Connect" button. After this, you should now be connected to your local runtime.

