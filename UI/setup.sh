#!/bin/bash

venv\Scripts\activate
deactivate

pip freeze > requirements.txt
pip install -r requirements.txt

python main.py


pyinstaller --onefile main.py

#my paths:
#C:\Users\avita\source\repos\ofekm5\ExtractingTurningPointGUI\t011222
#C:\Users\avita\source\repos\ofekm5\ExtractingTurningPointGUI\UI\utils\start_move_model.pth
#C:\Users\avita\source\repos\ofekm5\ExtractingTurningPointGUI\UI\utils\stop_turn_model.pth

```
