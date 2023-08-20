import random
import torch
from torch.utils.data import DataLoader
from torch import Tensor
import numpy as np
import yaml
import torch.nn.functional as F

from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models, ModelRegistryBase
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.modelbridge_utils import get_pending_observation_features
from ax.utils.testing.core_stubs import get_branin_search_space, get_branin_experiment
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.common.logger import get_logger

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.utils import standardize
from botorch.test_functions import BraninCurrin
from botorch.models.model import Model
from botorch.acquisition.input_constructors import acqf_input_constructor
from gpytorch.mlls import ExactMarginalLogLikelihood


from models import Donut_train, Donut_test, LSTM_train, LSTM_test, HW_train, HW_test
from models.Parameters import parameters
from utils.dataset import TZSDataset, TZSTestDataset
from utils.evaluate import best_f1_score_with_point_adjust
from utils.obj_fn import mse_obj, nf_obj
from utils.ac_fn import SimilarityWeightedExpectedImprovement


from typing import List, Any, Dict, Union


# setup_seed and logger
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
setup_seed(2023)
logger = get_logger(name="AutoKAD")
best_params = []


# Load configuration
with open("./conf.yml", 'r') as f:
    config = yaml.load(f, yaml.FullLoader)
device = config['device']


train_dataset = TZSDataset(config['train_kpi_path'])
test_dataset = TZSTestDataset(config['test_kpi_path'])


def evaluate(x_raw: np.ndarray, x_est: np.ndarray, labels: np.ndarray):
    anomaly_scores = np.abs(x_raw - x_est)
    mse = mse_obj(x_raw, x_est)
    nf = nf_obj(x_est)
    mse_nf = mse + nf

    res = best_f1_score_with_point_adjust(labels, anomaly_scores)

    res = {
        'precision': (res['p'], 0.0),
        'recall': (res['r'], 0.0),
        'f1': (res['r'], 0.0),
        'mse': (mse, 0.0),
        'nf': (nf, 0.0),
        'mse_nf': (mse_nf, 0.0)
    }

    return res


# Test dataset and labels are only used for experiment purpose.
def optimize_loop(params=None):
    model_name = params.get("model")

     # default for HW
    batch_size = 256
    win_len = 10

    # remove model tag:
    if model_name != 'HW':
        clear_params = {(i.replace('_'+model_name, "") if model_name in i else i): params[i] for i in params if i != "model"}
        clear_params['model'] = params['model']
        params = clear_params

        win_len = params.get('win_len')
        batch_size = params.get("batch_size")
    else:
        print("hlelo")
    logger.info(params)

    if model_name == "Donut":
        train = Donut_train
        test = Donut_test
    elif model_name == "LSTM":
        train = LSTM_train
        test = LSTM_test
    else:
        train = HW_train
        test = HW_test

    try:
        train_dataset.set_win_len(win_len)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        model = train(
            params=params, 
            dataloader=train_dataloader, 
            device=device
            )

        test_dataset.set_win_len(win_len)
        test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False)

        x_raw, x_est, loss, labels = test(
            model=model,
            dataloader=test_dataloader,
            device=device
        )

        res = evaluate(x_raw, x_est, labels)
        res['loss'] = (loss, 0.0)

    except Exception as e:
        print("Someting went wrong!")
        res = {
            'precision': (0.0, 0.0),
            'recall': (0.0, 0.0),
            'f1': (0.0, 0.0),
            'mse': (0x7fffffff, 0.0),
            'nf': (0x7fffffff, 0.0),
            'mse_nf': (0x7fffffff, 0.0),
        }

    # logger.info(f"performance: {res}")
    return res



@acqf_input_constructor(SimilarityWeightedExpectedImprovement)
def construct_inputs_scalarized_ucb(
    model: Model,
    best_f: Union[float, Tensor],
    simi_fn: callable,
    load_best_params,
    **kwargs: Any,
) -> Dict[str, Any]:
    return {
        "model": model,
        "best_f": torch.as_tensor(best_f, dtype=torch.float32),
        "simi_fn": simi_fn,
        "load_best_params": load_best_params
    }


gs = GenerationStrategy(
    steps=[
        # cold start step
        GenerationStep(
            model=Models.SOBOL,
            num_trials=3,  # How many trials should be produced from this generation step
            min_trials_observed=3, # How many trials need to be completed to move to next model
            max_parallelism=5,  # Max parallelism for this step
            model_kwargs={},  # Any kwargs you want passed into the model
            model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
        ),

        # Bayesian optimization step using the SW-EI acquisition function
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,  # No limitation on how many trials should be produced from this step
            # For `BOTORCH_MODULAR`, we pass in kwargs to specify what surrogate or acquisition function to use.
            # `acquisition_options` specifies the set of additional arguments to pass into the input constructor.
            model_kwargs={
                "botorch_acqf_class": SimilarityWeightedExpectedImprovement,
                "acquisition_options": {"best_f": 0.1, "simi_fn": F.cosine_similarity, "load_best_params": best_params},
            },
        ),
    ]
)


# Initialize the client - AxClient offers a convenient API to control the experiment
ax_client = AxClient(generation_strategy=gs)
# Setup the experiment
ax_client.create_experiment(
    name="AutoKAD_experiment",
    parameters=parameters,
    objectives={
        "mse_nf": ObjectiveProperties(minimize=True),
    },
    tracking_metric_names=["mse", "f1", "precision", "recall", "loss", "mse_nf", "nf"]
)
# Setup a function to evaluate the trials


for i in range(10):
    parameters, trial_index = ax_client.get_next_trial()
    best_params.append(parameters)
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=optimize_loop(parameters))

logger.info("Finished!")