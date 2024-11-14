import unittest
import os
import subprocess
import numpy as np
import importlib
import sashimi_si


class TestSashimi(unittest.TestCase):

    SASHIMI_INPUT = {
        'M0': 1.e12, 
        'redshift': 0., 
        'M0_at_redshift': True, 
        'dz': 0.01, 
        'N_herm': 20, 
        'zmax': 5., 
        'logmamin': 6, 
        'N_ma': 500
    }

    SASHMI_OUT_NAMES = names = ['ma200', 'z_acc', 'rsCDM_acc', 'rhosCDM_acc', 'rmaxCDM_acc', 'VmaxCDM_acc', 'rsSIDM_acc', 'rhosSIDM_acc', 'rcSIDM_acc', 'rmaxSIDM_acc', 'VmaxSIDM_acc', 'm_z0', 'rsCDM_z0', 'rhosCDM_z0', 'rmaxCDM_z0', 'VmaxCDM_z0', 'rsSIDM_z0', 'rhosSIDM_z0', 'rcSIDM_z0', 'rmaxSIDM_z0', 'VmaxSIDM_z0', 'ctCDM_z0', 'tt_ratio', 'weightCDM', 'weightSIDM', 'surviveCDM', 'surviveSIDM']

    COMMIT_ORIGINAL = "24d9895106640375eb74401da27a2a1ebed098ab"

    DIR = "log"

    def setUp(self) -> None:
        # get current branch by git branch --show-current
        self.branch_before_test = subprocess.check_output("git branch --show-current", shell=True).decode("utf-8").strip()
                # create directory if not exists
        if not os.path.exists(self.DIR):
            os.makedirs(self.DIR)
        # get the original sashimi output
        self.get_sashimi_output('original')
        return super().setUp()
    

    def tearDown(self) -> None:
        # return to the branch before the test
        status_code = os.system(f'git checkout {self.branch_before_test}')
        assert status_code == 0, f"failed to return to the branch {self.branch_before_test}"
        return super().tearDown()
    

    def get_sashimi_output(self, name):
        """ get sashimi output for given branch name.
        
        Args:
            name (str): branch name or commit hash. 
                - "local": load the sashimi on the current working directory.
                - "original": load the original sashimi.

        Returns:
            np.ndarray: sashimi output. shape=(N_parameters, N_simulations)
        """
        file = f"{self.DIR}/sashimi_output_{name}.npy"
        # check if the file exists and run the sashimi if not exists
        if not os.path.exists(file):
            if name == "local":
                print("loading local sashimi")
            else:
                self.name = self.COMMIT_ORIGINAL if name == 'original' else name
                cmd = f'git checkout {self.name}'
                print(f'{cmd}')
                status_code = os.system(cmd)
                assert status_code == 0, f"failed to checkout to the branch {name}"
            print(f'reloading sashimi_si({name}')
            importlib.reload(sashimi_si)
            # save sashimi output
            print(f'saving sashimi output to {file}')
            arr = (sashimi_si.subhalo_properties()
                    .subhalo_properties_calc(**self.SASHIMI_INPUT)
                    )
            # arr is a tuple of numpy arrays, (27, N_simulations)
            # convert it to a 2d numpy array, (27, N_simulations)
            arr = np.array(arr)
            print(f"output shape: {arr.shape}")
            assert arr.shape[0] == len(self.SASHMI_OUT_NAMES), f"output shape is different from the expected shape"
            np.save(file=file,arr=arr)
        return np.load(file)
    

    def compare_outputs(self, name1, name2):
        """ compare two sashimi outputs.

        Note that outputs have different shapes, so we only compare the values where weightCDM > 0.
        
        Args:
            name1 (str): branch name or commit hash. 
            name2 (str): branch name or commit hash. 

        Returns:
            self
        """
        idx_weight = self.SASHMI_OUT_NAMES.index('weightCDM')
        out1 = self.get_sashimi_output(name1)
        out2 = self.get_sashimi_output(name2)
        out1 = out1[:, out1[idx_weight] > 0]
        out2 = out2[:, out2[idx_weight] > 0]
        for name, _out1, _out2 in zip(self.SASHMI_OUT_NAMES, out1, out2):
            assert np.all(_out1 == _out2), f"output {name} is different between {name1} and {name2}"


    def test_memory_usage_reduction(self):
        self.compare_outputs('original', 'memory_usage_reduction')

    def test_parallelization(self):
        self.compare_outputs('original', 'parallelization')

    def test_local(self):
        self.compare_outputs('original', 'local')

    def test_main(self):
        self.compare_outputs('original', 'main')


