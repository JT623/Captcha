from conda/miniconda3-centos7
workdir /code
copy . /code
run pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt

