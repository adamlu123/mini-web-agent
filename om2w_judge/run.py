import argparse
import asyncio
import copy
import json
import multiprocessing
import os
import re
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from om2w_judge.methods.agenttrek_eval import AgentTrek_eval
from om2w_judge.methods.automomous_eval import Autonomous_eval
from om2w_judge.methods.webjudge_general_eval import WebJudge_general_eval
from om2w_judge.methods.webjudge_online_mind2web import WebJudge_Online_Mind2Web_eval
from om2w_judge.methods.webvoyager_eval import WebVoyager_eval
from om2w_judge.utils import OpenaiEngine, extract_predication


def auto_eval(args, task_subset, final_predicted_labels, lock, model):
    output_json_path = os.path.join(
        args.output_path,
        f"{args.mode}_{args.model}_score_threshold_{args.score_threshold}_auto_eval_results.json",
    )
    already_ids = []
    if os.path.exists(output_json_path):
        with open(output_json_path, "r", encoding="utf-8") as handle:
            already_data = handle.read()
        already_tasks = already_data.splitlines()
        for item in already_tasks:
            item = json.loads(item)
            already_ids.append(item["task_id"])

    print(f"The number of already done tasks: {len(already_ids)}")

    for task_id in task_subset:
        if task_id in already_ids:
            continue

        trajectory_images_path = os.path.join(args.trajectories_dir, task_id, "trajectory")
        screenshot_paths = []
        thoughts = None
        action_history = None
        final_result_response = None
        input_image_paths = None
        task_description = None

        with open(os.path.join(args.trajectories_dir, task_id, "result.json"), encoding="utf-8") as handle:
            result = json.load(handle)
            output_results = copy.deepcopy(result)
            task_description = result["task"]
            if "action_history" in result:
                action_history = result["action_history"]
            if "thoughts" in result:
                thoughts = result["thoughts"]
            if "final_result_response" in result:
                final_result_response = result["final_result_response"]
            if "input_image_paths" in result:
                input_image_paths = result["input_image_paths"]

        print(f"Start evaluation for {task_description}")
        if args.mode == "Autonomous_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg = Autonomous_eval(task_description, action_history, screenshot_paths[-1])
        elif args.mode == "AgentTrek_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg = AgentTrek_eval(task_description, action_history, thoughts, screenshot_paths[-1])
        elif args.mode == "WebVoyager_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg = WebVoyager_eval(task_description, screenshot_paths, final_result_response)
        elif args.mode == "WebJudge_Online_Mind2Web_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg, record, key_points = asyncio.run(
                WebJudge_Online_Mind2Web_eval(task_description, action_history, screenshot_paths, model, args.score_threshold)
            )
            output_results["image_judge_record"] = record
            output_results["key_points"] = key_points
        elif args.mode == "WebJudge_general_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg, record, key_points = asyncio.run(
                WebJudge_general_eval(
                    task_description,
                    input_image_paths,
                    thoughts,
                    action_history,
                    screenshot_paths,
                    model,
                    args.score_threshold,
                )
            )
            output_results["image_judge_record"] = record
            output_results["key_points"] = key_points
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        response = model.generate(messages, max_new_tokens=8192)[0]
        predicted_label = extract_predication(response, args.mode)

        evaluation_results = {"response": response, "predicted_label": predicted_label}
        output_results["task_id"] = task_id
        output_results["input_text"] = text
        output_results["system_msg"] = system_msg
        output_results["evaluation_details"] = evaluation_results
        output_results["predicted_label"] = predicted_label

        with lock:
            final_predicted_labels.append(predicted_label)

        print(f"Finish evaluation for {task_description}")
        print("=" * 20)
        os.makedirs(args.output_path, exist_ok=True)
        with lock:
            with open(output_json_path, "a+", encoding="utf-8") as handle:
                handle.write(json.dumps(output_results) + "\n")


def process_subset(task_subset, args, final_predicted_labels, lock):
    model = OpenaiEngine(model=args.model, api_key=args.api_key)
    auto_eval(args, task_subset, final_predicted_labels, lock, model)


def parallel_eval(args, num_workers=60):
    task_dirs = [
        directory
        for directory in sorted(os.listdir(args.trajectories_dir))
        if os.path.isdir(os.path.join(args.trajectories_dir, directory))
    ]
    print(f"Evaluating {len(task_dirs)} tasks in total.")
    if len(task_dirs) == 0:
        print("No tasks found.")
        return

    num_workers = max(1, min(num_workers, len(task_dirs)))
    task_subsets = [task_dirs[i::num_workers] for i in range(num_workers)]
    task_subsets = [subset for subset in task_subsets if subset]

    model = OpenaiEngine(model=args.model, api_key=args.api_key)

    if num_workers <= 1:
        lock = threading.Lock()
        final_predicted_labels = []
        auto_eval(args, task_dirs, final_predicted_labels, lock, model)
        success_num = sum(final_predicted_labels)
    else:
        lock = multiprocessing.Lock()
        with multiprocessing.Manager() as manager:
            final_predicted_labels = manager.list()
            processes = []
            for subset in task_subsets:
                process = multiprocessing.Process(
                    target=process_subset,
                    args=(subset, args, final_predicted_labels, lock),
                )
                process.start()
                processes.append(process)

            for process in processes:
                process.join()

            success_num = sum(final_predicted_labels)

    print("Evaluation complete.")
    print(f"The success rate is {(success_num / len(task_dirs)) * 100}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto evaluation of web navigation tasks.")
    parser.add_argument("--mode", type=str, default="Online_Mind2Web_eval", help="the mode of evaluation")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--trajectories_dir", type=str, required=True, help="Path to trajectories directory")
    parser.add_argument("--api_key", type=str, required=True, help="The api key")
    parser.add_argument("--output_path", type=str, required=True, help="The output path")
    parser.add_argument("--score_threshold", type=int, default=3)
    parser.add_argument("--num_worker", type=int, default=60)
    parser.add_argument("--num_proc", type=int, default=None, help="Number of worker processes (alias of --num_worker).")
    args = parser.parse_args()
    num_workers = args.num_proc if args.num_proc is not None else args.num_worker
    parallel_eval(args, num_workers)
