# MoA

## Install dependencies

```
pip install -r requirements.txt
```
You may encounter the problem while running LightGBM model:
```
libomp.dylib can't be loaded
```
This can be solved by installing `libomp` individually, for example on Mac
```
brew install libomp
```

## Run model

copy and paste training and test sets under `input`, then do
```
python run.py
```
