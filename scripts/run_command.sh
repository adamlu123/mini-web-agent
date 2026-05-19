source /home/luyadong/cred.sh
/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_iterative_judge.yaml \
    --workers 300 \
    --task-level all \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/iterative/0418_all_best_repro



source /home/luyadong/cred.sh
/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_iterative_judge_v2.yaml \
    --workers 300 \
    --task-level all \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/iterative/0419_all_best_v2


source /home/luyadong/cred.sh
/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_iterative_judge_v2.yaml \
    --workers 77 \
    --task-level hard \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/iterative/0421_hard_best_v3_write_judge_once


/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c iterative_strict.yaml \
    --workers 77 \
    --task-level hard \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/iterative/0421_hard_strict_v1


source /home/luyadong/cred.sh
/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_default_judge.yaml \
    --workers 77 \
    --task-level hard \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0421_hard_best_default_v1


source /home/luyadong/cred.sh
/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_default_judge_json.yaml \
    --workers 77 \
    --task-level hard \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0421_hard_best_default_json_sum20

/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_default_judge_json.yaml \
    --workers 300 \
    --task-level all \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0421_all_best_default_json_sum20_s300


/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_default_judge_json.yaml \
    --workers 300 \
    --task-level all \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0421_all_best_default_json_sum20_s300_final_resp



/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_default_judge_json_claude.yaml \
    --workers 100 \
    --task-level all \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0422_all_best_default_json_sum20_s300_final_resp_op47


/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_default_judge_json_kimi.yaml \
    --workers 10 \
    --task-level all \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0422_all_best_default_json_sum20_s300_final_resp_kimi25


/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_default_judge_json_kimi.yaml \
    --workers 10 \
    --task-level all \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0422_all_best_default_json_sum20_s300_final_resp_kimi26

/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_default_judge_json.yaml \
    --workers 77 \
    --task-level hard \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0423_hard_best_default_oracle



/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_default_judge_json.yaml \
    --workers 300 \
    --task-level all \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0424_all_v2


/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_default_judge_json.yaml \
    --workers 300 \
    --task-level all \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0425_all_addtask


/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_default_judge_json_oracle.yaml \
    --workers 77 \
    --task-level hard \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0425_real_oraclejudge


source /home/luyadong/cred.sh
python -m miniswewebagent.run.benchmarks.om2w \
  -c best_default_judge_json_openrouter_qwen36_cli_multi.yaml \
  --tasks-file /home/luyadong/sandbox/mini-web-agent/src/miniswewebagent/run/benchmarks/om2w_260220_multi_cli.json \
  --workers 209 \
  --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0428_qwen35_9b_multi_cli \
  --no-evaluate



/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
    -c best_default_judge_json_kimi.yaml \
    --workers 15 \
    --task-level hard \
    --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0429/0429_hard_best_default_json_sum20_s300_final_resp_kimi25




/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
  -c best_default_judge_json.yaml \
  --workers 200 \
  --tasks-file /home/luyadong/sandbox/mini-web-agent/src/miniswewebagent/run/benchmarks/odysseys.json \
  --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0429/0429_odysseys_best_default_json_s100 


/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
  -c best_default_judge_json.yaml \
  --workers 200 \
  --tasks-file /home/luyadong/sandbox/mini-web-agent/src/miniswewebagent/run/benchmarks/odysseys.json \
  --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0429/0429_odysseys_best_default_json_s100_r2 


/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
  -c best_default_judge_json.yaml \
  --workers 200 \
  --tasks-file /home/luyadong/sandbox/mini-web-agent/src/miniswewebagent/run/benchmarks/odysseys.json \
  --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0429/0429_odysseys_fixbadimg


/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
  -c best_default_judge_json.yaml \
  --workers 200 \
  --tasks-file /home/luyadong/sandbox/mini-web-agent/src/miniswewebagent/run/benchmarks/odysseys.json \
  --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0429/0429_odysseys_fixbadimg_s20


/home/luyadong/.venv/bin/python -m miniswewebagent.run.benchmarks.om2w \
  -c best_default_judge_json.yaml \
  --workers 200 \
  --tasks-file /home/luyadong/sandbox/mini-web-agent/src/miniswewebagent/run/benchmarks/odysseys.json \
  --output-dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0430/0430_odysseys
