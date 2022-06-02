# Package Description

A python webcrawler with NLP modules as optional custom built using beautiful soup and many open source packages. it is purpose built with its custom data structure to hold maximum amount of data while maintaining ease of use and access of data.

### Streamlit version:
A purpose built streamlit frontend which can guide user through all steps in generating the final data.
___
## Pre-requisites
#### Common dependency:
- JDK and JAVA_HOME variable set
- apache maven : https://maven.apache.org/install.html
- install sutime before installing package(https://github.com/FraBle/python-sutime):
    ```bash
    >> # Ideally, create a virtual environment before installing any dependencies
    >> pip install sutime
    >> # Install Java dependencies
    >> mvn dependency:copy-dependencies -DoutputDirectory=./jars -f $(python3 -c 'import importlib; import pathlib; print(pathlib.Path(importlib.util.find_spec("sutime").origin).parent / "pom.xml")')
    ```
 #### Streamlit app dependency:
 ```bash
 pip install streamlit
 pip install streamlit-aggrid
 pip install streamlit-multipage
 pip install requests_cache
 ```

## Installation
```bash
pip install pkg_name
```

## Usage
#### Py-Pkg:
```python
from crawler_pkg import crawl

# start crawling
crawl.run(url="")
```
#### streamlit app:
```bash
streamlit run engine.py
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## LICENSE
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/rick-cee-137/crawler/blob/a4a0d25e1dea1201b9f98938d4f51618a3ab4058/LICENSE)

_____
![](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)
![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)


![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

	
