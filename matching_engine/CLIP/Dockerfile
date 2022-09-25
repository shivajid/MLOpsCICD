FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN apt-get install -y --no-install-recommends wget
RUN apt-get -y update
RUN apt-get -y install git
RUN pip install ftfy regex tqdm
RUN pip install git+https://github.com/openai/CLIP.git

LABEL "com.example.vendor"="SHIVAJI Incorporated"
LABEL com.example.label-with-value="foo"
LABEL version="1.0"
LABEL description="This text illustrates \
that label-values can span multiple lines."

RUN mkdir /working_dir

COPY . /working_dir/

#CMD ["python3","/working_dir/sample.py"]
CMD python3 /working_dir/sample.py 


