import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from avalanche.evaluation import metrics
from models import MultiHeadMLP
from experiments.skill_memory import SkillMemoryPlugin
from experiments.utils import set_seed, create_default_args


def skill_memory_smnist(override_args=None):
    """Skill Memory on Split MNIST.

    The plugin stores acquired model states and, for each new experience,
    chooses between reusing a compatible skill, cloning it and continuing
    training, or starting from the initial model state.
    """
    args = create_default_args(
        {
            'cuda': 0,
            'epochs': 10,
            'learning_rate': 0.001,
            'train_mb_size': 64,
            'seed': None,
            'reuse_threshold': 0.90,
            'clone_threshold': 0.30,
            'forgetting_margin': 0.05,
            'probe_samples': 64,
            'probe_batches': 5,
        },
        override_args,
    )
    set_seed(args.seed)
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    benchmark = avl.benchmarks.SplitMNIST(
        5, return_task_id=True, fixed_class_order=list(range(10))
    )
    model = MultiHeadMLP(hidden_size=256, hidden_layers=2)
    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()
    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(experience=True, stream=True),
        loggers=[interactive_logger],
    )

    skill_memory = SkillMemoryPlugin(
        reuse_threshold=args.reuse_threshold,
        clone_threshold=args.clone_threshold,
        forgetting_margin=args.forgetting_margin,
        probe_samples=args.probe_samples,
        probe_batches=args.probe_batches,
        probe_seed=args.seed or 0,
    )

    cl_strategy = avl.training.Naive(
        model,
        Adam(model.parameters(), lr=args.learning_rate),
        criterion,
        train_mb_size=args.train_mb_size,
        train_epochs=args.epochs,
        eval_mb_size=128,
        device=device,
        evaluator=evaluation_plugin,
        plugins=[skill_memory],
    )

    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        cl_strategy.eval(benchmark.test_stream)

    return cl_strategy.eval(benchmark.test_stream)


if __name__ == '__main__':
    res = skill_memory_smnist()
    print(res)
