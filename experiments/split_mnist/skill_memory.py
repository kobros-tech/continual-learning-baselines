import argparse

import avalanche as avl
import torch
from avalanche.evaluation import metrics
from torch.nn import CrossEntropyLoss

from experiments.utils import create_default_args, set_seed
from skill_memory import SkillMemoryStrategy


def skill_memory_smnist(override_args=None):
    args = create_default_args(
        {
            "cuda": 0,
            "epochs": 1,
            "learning_rate": 0.01,
            "train_mb_size": 64,
            "seed": 0,
            "dataset_dir": None,
            "max_skills": 20,
            "forgetting_margin": 0.05,
            "score_floor": 0.9,
            "probe_batch_size": 64,
            "probe_batches": 5,
            "probe_seed": 0,
            "class_train_batch_size": 64,
            "reuse_is_mutable": True,
            "eval_memory_per_class": 200,
            "eval_memory_seed": 0,
            "eval_epochs": 3,
            "eval_batch_size": 64,
            "eval_learning_rate": 0.01,
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
        5,
        return_task_id=False,
        fixed_class_order=list(range(10)),
        seed=args.seed,
        dataset_root=args.dataset_dir,
    )
    model = avl.models.SimpleMLP(num_classes=10)
    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()
    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger],
    )

    strategy = SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
        ),
        criterion=criterion,
        evaluator=evaluation_plugin,
        max_skills=args.max_skills,
        forgetting_margin=args.forgetting_margin,
        score_floor=args.score_floor,
        probe_batch_size=args.probe_batch_size,
        probe_batches=args.probe_batches,
        probe_seed=args.probe_seed,
        class_train_batch_size=args.class_train_batch_size,
        reuse_is_mutable=args.reuse_is_mutable,
        eval_memory_per_class=args.eval_memory_per_class,
        eval_memory_seed=args.eval_memory_seed,
        eval_epochs=args.eval_epochs,
        eval_batch_size=args.eval_batch_size,
        eval_learning_rate=args.eval_learning_rate,
        evaluator_model_factory=lambda: avl.models.SimpleMLP(
            num_classes=10,
            input_size=28 * 28,
            hidden_size=2048, 
            hidden_layers=1,
            drop_rate=0.1,
        ),
        train_mb_size=args.train_mb_size,
        train_epochs=args.epochs,
        eval_mb_size=128,
        device=device,
        eval_routing="none",
    )

    res = None
    for experience in benchmark.train_stream:
        strategy.train(experience)
        res = strategy.eval(benchmark.test_stream)

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Root directory containing the MNIST dataset.",
    )
    args = parser.parse_args()

    print(skill_memory_smnist({"dataset_dir": args.dataset_dir}))
