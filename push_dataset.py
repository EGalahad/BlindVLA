from lerobot.datasets.lerobot_dataset import LeRobotDataset

root= "lerobot_out/blindvla_1k_lerobot"
repo_id= "elijahgalahad/blindvla_1k_lerobot"
dataset = LeRobotDataset(repo_id=repo_id, root=root)
dataset.push_to_hub(upload_large_folder=True)