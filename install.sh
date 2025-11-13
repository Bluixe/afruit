# conda create --name afruit python=3.11 -y
# conda activate afruit

pip install torch torchvision

pip install -r requirements.txt

pip install -e .

cd afruits/stable-baselines3

pip install -e .

pip install tianshou

pip install PyQt5