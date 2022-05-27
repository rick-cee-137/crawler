# Package Description

A python webcrawler with NLP modules as optional custom built using beautiful soup and many open source packages. it is purpose built with its custom data structure to hold maximum amount of data while maintaining ease of use and access of data.

## Pre-requisites

- JDK and JAVA_HOME variable set
- apache maven : https://maven.apache.org/install.html
- install sutime before installing package(https://github.com/FraBle/python-sutime):
    ```bash
    >> # Ideally, create a virtual environment before installing any dependencies
    >> pip install sutime
    >> # Install Java dependencies
    >> mvn dependency:copy-dependencies -DoutputDirectory=./jars -f $(python3 -c 'import importlib; import pathlib; print(pathlib.Path(importlib.util.find_spec("sutime").origin).parent / "pom.xml")')
    ```

## Installation
```bash
pip install pkg_name
```

## Usage

```python
from crawler_pkg import crawl

# start crawling
crawl.run(url="")
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
