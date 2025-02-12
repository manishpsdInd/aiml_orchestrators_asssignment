## AI ML Orchestrators Assignment
Handwritten digit recognition implement enable model to classify handwritten digits from the MNIST dataset.

<h3 align="center">Passionate developers </h3>
<p align="left"> Sai Prasad | prasad.ravinuthala@gmail.com</p>
<p align="left"> Manish Prasad | manishpsd@gmail.com</p>

<h4 align="left">Languages and Tools:</h4>
<p align="left"> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> </p>

```
project-root/
├── .venv/                # Virtual environment
├── config/               # Config files
│   ├── dev_config.yaml
│   ├── prod_config.yaml
├── data/                 # Input/output data
│   ├── input/
│   ├── processed/
│   ├── output/
├── docs/                 # Documentation
│   ├── api_reference.md
│   ├── setup_guide.md
├── logs/                 # Log files
│   └── app.log
├── scripts/              # Build and deploy scripts
│   ├── build/
│   │   └── genssl.sh
│   ├── deploy/
│   │   └── application.sh
├── src/                  # Source code
│   ├── core/
│   │   ├── __init__.py
│   │   ├── parser.py
│   │   ├── mapper.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── helpers.py
│   ├── __init__.py
│   ├── application.py
│   └── main.py
├── tests/                # Tests
│   ├── unit/
│   ├── integration/
│   ├── mocks/
│   ├── __init__.py
│   ├── test_main.py
├── pyproject.toml        # Project dependencies and settings
├── README.md             # Project documentation
├── setup.py              # Package setup script
└── requirements.txt      # Python package dependencies (optional)
```

#### Git Link:
https://github.com/manishpsdInd/aiml_orchestrators_asssignment.git

-------------

docker build -t aiml_orchestrators_asssignment .

docker login -u manishpsd

docker images

docker tag aiml_orchestrators_asssignment:latest manishpsd/aiml_orchestrators_asssignment:latest

docker push manishpsd/aiml_orchestrators_asssignment:latest

-------------

docker pull manishpsd/aiml_orchestrators_asssignment:latest

docker run -p 8080:8080 manishpsd/aiml_orchestrators_asssignment

-------------

https://hub.docker.com/r/manishpsd/aiml_orchestrators_asssignment

-------------