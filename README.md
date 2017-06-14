# fakepark

Simulating fake electrophysiological features.

# Install

`git clone git@github.com:brainprosthesis/fakephys.git` 

onto your Python path. (`pip install .` coming in Beta 2).


# Usage

For interactive use see the notebooks in `ipynb/`.

For use of the `.yaml` data design format (which is run from the command line) git clone as above, then run:

    python fakephys/run.py --name=demo simulate --config=fakephys/demo.yaml --save
   
  
# Dependencies

    - sdeint
    - numpy
    - scipy
    - brian2
