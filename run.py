import fire
import numpy as np
from fakephys import model, noise

from yaml import load, dump
try:
    from pyaml import dump as pdump
except ImportError:
    print("pyaml not installed. Using PyYAML instead. "
          "For more readable conversion output install pyaml.")
    pdump = dump


class Run(object):
    def __init__(self, name=""):
        # Set by asst. methods
        self._name = name
        self._X = None
        self._conds = None
        self._n_conds = None

        self._t = None
        self._dt = None
        self._n_electrodes = None

    def simulate(self, config=None, save=False):
        # Load
        with open(config, 'r') as f:
            d = load(f)

        if d is None:
            raise ValueError("{} was empty.".format(config))

        # Get globals.
        params = d["shared_parameters"]
        n_electrodes = params['n_electrodes']
        t = params['t']
        dt = params['dt']

        # Shared noise? (For use after model runs).
        try:
            shared_noise = d["shared_noise"]
        except KeyError:
            noises = None

        # Init X, the final dataset
        data = d["data"]
        conds = sorted(data.keys())
        n_conds = len(conds)

        # Generate the data
        X = []
        for c in conds:
            Xc = []
            models_c = data[c]
            for m, p in models_c.items():
                # n needs to be a pos. arg.
                n = p["n"]
                del p["n"]

                # set dt for kwargs unpacking
                p.update({"dt": dt})

                # model
                Xc.append(getattr(model, m)(t, n, **p))

            # Transpose for code–based stacking at the end
            Xc = np.vstack(Xc)
            X.append(np.copy(Xc))

        # Add noise (independently for each code (def above))
        if shared_noise is not None:
            index = sorted(shared_noise.keys())
            for i in index:
                noise_i = shared_noise[i]
                for m, p in noise_i.items():
                    # set dt for kwargs unpacking
                    p.update({"dt": dt})

                    # Add noise splitting by conds
                    for k in range(n_conds):
                        X[k] = getattr(noise, m)(X[k], **p)

        # Store data
        self._X = X

        # store some key config values, for convenience
        self._n_electrodes = n_electrodes
        self._t = t
        self._dt = dt
        self._conds = conds
        self._n_conds = n_conds

        if save:
            self._save()

    def _save(self):
        name = self._name
        for k, c in enumerate(self._conds):
            np.savetxt(name + "_{}.csv".format(c), self._X[k], delimiter=',')


if __name__ == "__main__":
    fire.Fire(Run)  # Define the CL interface
