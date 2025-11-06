# conda create --name afruit python=3.11 -y
# conda activate afruit

pip install torch torchvision

pip install -r requirements.txt

pip install -e .

cd stable-baselines3

pip install -e .

cd ../tianshou

pip install -e .

cd ..