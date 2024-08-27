#!/bin/bash

venv\Scripts\activate
deactivate

pip freeze > requirements.txt
pip install -r requirements.txt

python main.py


pyinstaller --onefile main.py

```
