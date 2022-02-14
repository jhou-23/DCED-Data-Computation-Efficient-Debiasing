import argparse
import json
import os

import transformers

from stereoset_utils import StereoSetRunner
import models

def generate_experiment_id(
    name,
    model=None,
    intrasentence_model=None,
    model_name_or_path=None,
    bias_type=None,
    score_type=None,
    data_split=None,
    representation_type=None,
):
    experiment_id = f"{name}"

    # Build the experiment ID.
    if isinstance(model, str):
        experiment_id += f"_m-{model}"
    if isinstance(intrasentence_model, str):
        experiment_id += f"_intra-{intrasentence_model}"
    if isinstance(model_name_or_path, str):
        experiment_id += f"_c-{model_name_or_path}"
    if isinstance(bias_type, str):
        experiment_id += f"_t-{bias_type}"
    if isinstance(score_type, str):
        experiment_id += f"_s-{score_type}"
    if isinstance(data_split, str):
        experiment_id += f"_d-{data_split}"
    if isinstance(representation_type, str):
        experiment_id += f"_r-{representation_type}"

    return experiment_id

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs StereoSet benchmark.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir)),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--intrasentence_model",
    action="store",
    type=str,
    default="BertForMaskedLM",
    choices=[
        "BertForMaskedLM",
        "AlbertForMaskedLM",
    ],
    help="Model to evalute (e.g., BertModel). Typically, these correspond to a HuggingFace "
    "class.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--score_type",
    action="store",
    type=str,
    default="likelihood",
    choices=["likelihood"],
    help="The StereoSet scoring mechanism to use.",
)
parser.add_argument(
    "--split",
    action="store",
    type=str,
    default="dev",
    choices=["dev", "test"],
    help="The StereoSet split to use.",
)
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=1,
    help="The batch size to use during StereoSet intrasentence evaluation.",
)
parser.add_argument("--adapter", action="store", type=str, default=None, help="Adapter")


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="stereoset",
        intrasentence_model=args.intrasentence_model,
        model_name_or_path=args.model_name_or_path,
        score_type=args.score_type,
        data_split=args.split,
    )

    print("Running StereoSet:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - intrasentence_model: {args.intrasentence_model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - score_type: {args.score_type}")
    print(f" - split: {args.split}")
    print(f" - batch_size: {args.batch_size}")

    intrasentence_model = getattr(models, args.intrasentence_model).from_pretrained(args.model_name_or_path)
    intrasentence_model.eval()

    if args.adapter != None:
        intrasentence_model.set_active_adapters("cda")

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased"
        # args.model_name_or_path, add_prefix_space=add_prefix_space
    )

    runner = StereoSetRunner(
        intrasentence_model=intrasentence_model,
        tokenizer=tokenizer,
        input_file=f"{args.persistent_dir}/data/stereoset/{args.split}.json",
        model_name_or_path=args.model_name_or_path,
        score_type=args.score_type,
        batch_size=args.batch_size
    )
    results = runner()

    os.makedirs(f"{args.persistent_dir}/results/stereoset", exist_ok=True)
    with open(
        f"{args.persistent_dir}/results/stereoset/{experiment_id}.json", "w"
    ) as f:
        json.dump(results, f, indent=2)