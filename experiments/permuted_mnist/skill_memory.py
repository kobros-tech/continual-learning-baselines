import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from avalanche.evaluation import metrics
from models import MLP
from experiments.skill_memory import SkillMemoryPlugin
from experiments.utils import set_seed, create_default_args


def skill_memory_pmnist(override_args=None):
    """Experimental Skill Memory baseline on Permuted MNIST."""
    args = create_default_args(
        {
            'cuda': 0,
            'epochs': 10,
            'learning_rate': 0.001,
            'train_mb_size': 256,
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

    benchmark = avl.benchmarks.PermutedMNIST(10)
    model = MLP(hidden_size=256, hidden_layers=2, output_size=10)
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

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = skill_memory_pmnist()
    print(res)
