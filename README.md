# fakephys

Simulating fake electrophysiological features.

# Install

Assuming you're at the command line, clone from github.

        git clone git@github.com:brainprosthesis/fakephys.git

Then move to the cloned directory.

        cd fakephys

And finally type.

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
