# fakephys

Simulating fake electrophysiological features.

# Install

First clone from gitub

        git clone git@github.com:brainprosthesis/fakephys.git

The move to the cloned directory, on unix from the command line

        cd fakephys

And finally from the command line type

        pip install .


# Usage

For interactive use see the notebooks in `ipynb/`.

For use of the `.yaml` data design format install as above, then use the `phys` command line interface. For example,

    phys --name=demo simulate --config=fakephys/demo.yaml --save
   
    
# Dependencies

    - sdeint
    - numpy
    - scipy
    - brian2
